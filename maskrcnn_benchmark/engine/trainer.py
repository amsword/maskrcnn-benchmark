# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import time

import torch
import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size, get_rank
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        name_dims = [(k, v.dim()) for k, v in zip(loss_names, all_losses)]
        for k in range(1, len(name_dims)):
            if name_dims[k][1] != name_dims[0][1]:
                logging.info('{}={} not equal {}={}'.format(name_dims[k][0],
                    name_dims[k][1], name_dims[0][0], name_dims[0][1]))
                raise Exception()
        all_losses = torch.stack(all_losses, dim=0)
        from qd.qd_common import is_hvd_initialized
        if not is_hvd_initialized():
            dist.reduce(all_losses, dst=0)
        else:
            import horovod.torch as hvd
            all_losses = hvd.allreduce(all_losses, average=False)
        if get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses

def forward_backward(model, images, targets,
        optimizer,
        arguments, checkpointer, use_hvd,
        meters, device, loss_scalar, no_update=False):
    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values()) * loss_scalar
    if losses != losses:
        logging.info('NaN encountered!')
        arguments['images'] = images
        arguments['targets'] = targets
        checkpointer.save("NaN_context_{}".format(get_rank()), **arguments)
        raise RuntimeError('NaN encountered!')

    # reduce losses over all GPUs for logging purposes
    if not use_hvd:
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)
    else:
        losses_reduced = sum(loss for loss in loss_dict.values())
        meters.update(loss=losses_reduced, **loss_dict)

    # Note: If mixed precision is not used, this ends up doing nothing
    # Otherwise apply loss scaling for mixed-precision recipe
    if not no_update:
       if device.type == 'cpu':
           losses.backward()
       else:
           if not use_hvd:
               from apex import amp
               with amp.scale_loss(losses, optimizer) as scaled_losses:
                   scaled_losses.backward()
           else:
               losses.backward()

def partition_data(images, targets, num):
    if num == 1 or len(images.image_sizes) < num:
        return [(images, targets)]
    each = len(images.image_sizes) // num
    result = []
    from maskrcnn_benchmark.structures.image_list import ImageList
    for i in range(num):
        start = i * each
        end = start + each
        curr_tensors = images.tensors[start: end]
        curr_sizes = images.image_sizes[start: end]
        curr_imagelist = ImageList(curr_tensors, curr_sizes)
        curr_target = targets[start: end]
        result.append((curr_imagelist, curr_target))
    return result

def average_gradients(model):
    size = dist.get_world_size()
    if size == 1:
        return
    size = float(size)
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            param.grad.data /= size

from qd.qd_common import try_once
@try_once
def try_save_intermediate_snapshot(checkpointer, iteration, arguments):
    checkpointer.save("model_{:07d}".format(iteration), **arguments)

def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    arguments,
    log_step=20,
    data_partition=1,
    explicit_average_grad=False,
    set_model_to_train=True,
    no_update=False,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    # Set model to train before freezing parameters, because BN needs to be
    # eval mode
    if set_model_to_train:
        model.train()
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    from qd.qd_common import is_hvd_initialized
    use_hvd = is_hvd_initialized()
    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        if hasattr(images, 'image_sizes') and len(images.image_sizes) == 0:
            logging.error('this should never happen since different workers '
                    'will have different numbers of iterations.')
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        if not no_update:
            scheduler.step()

        if isinstance(images, list):
            images = [x.to(device )for x in images]
        else:
            images = images.to(device)
        if isinstance(targets, torch.Tensor):
            targets = targets.to(device)
        else:
            targets = [target.to(device) for target in targets]

        if not no_update:
            optimizer.zero_grad()

        all_image_target = partition_data(images,
                targets, data_partition)

        for curr_images, curr_target in all_image_target:
            forward_backward(model, curr_images, curr_target,
                    optimizer,
                    arguments, checkpointer, use_hvd,
                    meters, device, loss_scalar=1./data_partition,
                    no_update=no_update)
        if explicit_average_grad:
            average_gradients(model)

        if not no_update:
            optimizer.step()

        batch_time = time.time() - end
        end = time.time()

        if iteration > start_iter + 5:
            # we will skip the first few iterations since the time cost
            # evaluation for those are not good
            meters.update(time=batch_time, data=data_time)

        if iteration % log_step == 0 or iteration == max_iter:
            speed = get_world_size() * log_step * len(targets) / (time.time() - log_start)
            if hasattr(meters, 'time'):
                eta_seconds = meters.time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            else:
                eta_string = 'Unknown'

            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        'speed: {speed:.1f} images/sec',
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    speed=speed,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            log_start = time.time()
        if iteration % checkpoint_period == 0:
            # with blobfuse, saving could fail with unknown reason. Instead of
            # saving and crashing, we do a best-effort manner.
            try_save_intermediate_snapshot(checkpointer, iteration, arguments)
        if iteration >= max_iter:
            checkpointer.save("model_final", **arguments)
            if get_rank() > 0:
                old_value = checkpointer.save_to_disk
                checkpointer.save_to_disk = True
                checkpointer.save("model_final_{}".format(get_rank()), **arguments)
                checkpointer.save_to_disk = old_value

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
