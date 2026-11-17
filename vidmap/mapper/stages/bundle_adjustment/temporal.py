"""BA temporal smoothing on world-space rig centres."""

import pyceres
from vidmap_native import bundle_adjustment as ba_costs
from vidmap_native import global_positioning as gp_costs


def append_temporal_priors(problem, reconstruction, config, options, priors, temporal):
    """Reuse GP triplets and the keypoint budget throughout every pose-changing BA."""
    if not temporal.enabled or temporal.weight == 0 or not options.refine_rig_from_world:
        return [], set()
    active = {reconstruction.image(i).frame_id for i in config.images}
    owners = []
    constrained_frames = set()
    for prior in priors:
        ids = (prior.prev_frame_id, prior.frame_id, prior.next_frame_id)
        if not all(fid in active for fid in ids):
            continue
        loss = prior.loss(
            stddev=temporal.stddev,
            dead_zone=0.0,
            huber_width=temporal.huber_width,
            relative_dead_zone=temporal.relative_dead_zone,
        )
        poses = [reconstruction.frame(fid).rig_from_world.params for fid in ids]
        new = [not problem.has_parameter_block(p) for p in poses]
        cost = ba_costs.temporal_acceleration_cost(prior.dt_prev, prior.dt_next, temporal.stddev)
        weighted = gp_costs.scaled_loss(loss, temporal.weight * prior.sqrt_point_budget**2)
        problem.add_residual_block(cost, weighted, poses)
        constrained_frames.update(ids)
        owners.extend((cost, loss, weighted, *poses))
        for fid, pose, added in zip(ids, poses, new):
            if not added:
                continue
            if config.has_constant_rig_from_world_pose(fid):
                problem.set_parameter_block_constant(pose)
            else:
                # No observation constrains this pose's orientation. Preserve RA.
                manifold = pyceres.SubsetManifold(7, [0, 1, 2, 3])
                problem.set_manifold(pose, manifold)
                owners.append(manifold)
    return owners, constrained_frames
