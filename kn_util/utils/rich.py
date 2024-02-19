from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    DownloadColumn,
    TransferSpeedColumn,
    Console,
)


def get_rich_progress_mofn(text_format="[bold blue]{task.description}", **kwargs):
    progress = Progress(
        # TextColumn("[bold blue]{task.fields[name]}", justify="right"),
        TextColumn(text_format, justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        MofNCompleteColumn(),
        console=Console(record=True),
        **kwargs,
    )
    return progress


def get_rich_progress_download(text_format="[bold blue]{task.description}", **kwargs):
    progress = Progress(
        TextColumn(text_format, justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=Console(record=False),
        **kwargs,
    )
    return progress


def add_tasks(progress, names, totals):
    task_ids = []
    for name, total in zip(names, totals):
        task_id = progress.add_task(name, total=total)
        task_ids.append(task_id)
    return task_ids
