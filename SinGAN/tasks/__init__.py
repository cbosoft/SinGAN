from .train import run_task_train
from .harmonisation import run_task_harmonisation

task_list = [lv[9:] for lv in locals().keys() if lv.startswith('run_task_')]


def run_task(cfg):
    fmt_task_list = ', '.join(task_list[:-1]) + ', or ' + task_list[-1]
    assert cfg.mode in task_list, f'Unrecognised task: "{cfg.mode}". Must be one of {fmt_task_list}'

    print(f'Running task: {cfg.mode}')

    # get task func from global variables and run it
    globals()[f'run_task_{cfg.mode}'](cfg)
