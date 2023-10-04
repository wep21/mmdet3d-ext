import click
from tools.mono_nuscenes_converter import GTDatabaseCreater


@click.command()
@click.option('--data_path', type=str)
@click.option('--dataset_ids', '-d', multiple=True, default=())
@click.option('--lidar_type', type=str)
@click.option('--num_worker', type=int, default=8)
def cli(
    data_path: str,
    dataset_ids: tuple[str, ...],
    lidar_type: str,
    num_worker: int,
) -> None:
    GTDatabaseCreater(
        data_path=data_path,
        dataset_ids=list(dataset_ids),
        lidar_type=lidar_type,
        num_worker=num_worker,
    ).create()


if __name__ == '__main__':
    cli()