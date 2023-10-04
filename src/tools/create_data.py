import click
from tools.mono_nuscenes_converter import create_mono_nuscenes_infos


@click.command(
    context_settings=dict(ignore_unknown_options=True),
)
@click.option('--root_path', type=str)
@click.option('--dataset_id', type=str)
@click.option('--info_prefix', type=str, default='nuscenes')
@click.option('--lidar_type', type=str, default='LIDAR_TOP')
@click.option('--camera_types', '-c', multiple=True, default=(
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ))
@click.option('--max_sweeps', type=int, default=10)
def cli(
    root_path: str,
    dataset_id: str,
    info_prefix: str,
    lidar_type: str,
    camera_types: tuple[str, ...],
    max_sweeps: int = 10,
) -> None:
    create_mono_nuscenes_infos(
        root_path=root_path,
        dataset_id=dataset_id,
        info_prefix=info_prefix,
        lidar_type=lidar_type,
        camera_types=list(camera_types),
        max_sweeps=max_sweeps,
    )


if __name__ == '__main__':
    cli()