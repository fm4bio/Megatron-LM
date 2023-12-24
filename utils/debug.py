import torch
import traceback
import importlib
import re
import os
import pandas as pd

WORLD_RANK = int(os.environ.get("WORLD_RANK", "0"))  # can be set by train.py

############ storing training loss ############
# def _update_loss_df(df, meta, loss):
#     record = {**meta, **loss}
#     if len(df) == 0:
#         df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
#         return df

#     # find the existing value in the dataframe
#     selector = (df["pdb_path"] == meta["pdb_path"]) & (df["chain_id"] == meta["chain_id"]) & (df["in_dataset_idx"] == meta["in_dataset_idx"])
#     row = df.loc[selector]
#     if len(row) == 0:
#         df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
#     else:
#         # update the existing value
#         for k, v in loss.items():
#             df.loc[selector, k] = v
#     return df


# def store_loss_to_df(meta, loss_dict):
#     global LOSS_DF
#     loss_dict = {k: float(v) for k, v in loss_dict.items()}
#     LOSS_DF = _update_loss_df(LOSS_DF, meta, loss_dict)


# def save_loss_df(out_csv):
#     LOSS_DF.to_csv(out_csv, index=False)


# DEBUG_STORE_LOSS = bool(os.environ.get("DEBUG_STORE_LOSS", 1))
# IS_DEBUG_STORE_LOSS = lambda: DEBUG_STORE_LOSS and WORLD_RANK == 0
# LOSS_DF = pd.DataFrame()
########### end of storing training loss ###########


def return_default_when_fail(func, default):
    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            return ret
        except Exception as e:
            print(e)
            traceback.print_exc()
            return default(*args, **kwargs)

    return wrapper


def save_input_when_fail(func):
    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            return ret
        except Exception as e:
            try:
                torch.save((args, kwargs), "crash.input")
            except Exception as e2:  # when self is in args, sometimes it can't be dumped
                print(e2)
                traceback.print_exc()
                torch.save((args[1:], kwargs), "crash.input")
            raise e

    return wrapper


def print_input_when_fail(func):
    def wrapper(*args, **kwargs):
        try:
            ret = func(*args, **kwargs)
            return ret
        except Exception as e:
            try:
                print("function =", func.__name__)
                print("args =", args)
                print("kwargs =", kwargs)
            except Exception as e2:  # when self is in args, sometimes it can't be dumped
                print(e2)
                traceback.print_exc()
                torch.save((args[1:], kwargs), "crash.input")
            raise e

    return wrapper


def reload(module):
    return importlib.reload(module)


debug_input_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_inputs")
ignored_interactive = set()


def interactive(title=""):
    if title in ignored_interactive:
        return ""
    print(f"{title}")
    # to access the local variables, we return the following executable string. you need to call exec by your self.
    # from ml_tools.debug import interactive; exec(interactive())
    return rf"""
print("enter debug: {title}")
import traceback
import re

while True:
    try:
        inp = input("\r> ")

        # special cases
        if inp.startswith("`"):
            print("load from file")
            if len(inp) == 1:
                file = "{debug_input_path}/input_1.py"
            elif re.match("\d+", inp[1:]):
                file = "{debug_input_path}/input_" + inp[1:] + ".py"
            else:
                file = inp[1:].strip()
            try:
                inp = "".join(open(file).readlines())
                print(inp)
            except Exception as e:
                traceback.print_exc()
                print("load debug_input failed")
                continue

        elif inp.startswith("SKIP"):
            try:
                import ml_tools.debug as debug
                title = re.match("SKIP (.+)", inp).group(1)
                debug.ignored_interactive.add(title)
                print(f"skip command succeeded: added " + title + " to ignored")
                continue
            except Exception as e:
                traceback.print_exc()
                print("skip command failed. Usage: SKIP <title>")
                continue

        # execute the input
        try:
            print(eval(inp))
        except Exception as e:
            try:
                exec(inp)
            except Exception as e2:
                traceback.print_exc()
                print("eval:", e)
                print("exec:", e2)
    except (KeyboardInterrupt, EOFError):
        print()
        print("exit debug")
        break"""


if __name__ == "__main__":
    exec(interactive("interactive"))
