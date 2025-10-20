def next_block_prompt(ix_next_block):
    question_exec = ""
    while not ("y" in question_exec or "n" in question_exec):
        question_exec = input(f"Execute next block (block {ix_next_block})? [y/n]")

    return "y" in question_exec


def stim_prompt(stim, ix_next_block):
    if stim is not None:
        return stim
    else:
        question_stim = ""
        while not (question_stim in ["on", "off"]):
            question_stim = input(f"Stim for (block {ix_next_block})? [on/off]")
        return question_stim
