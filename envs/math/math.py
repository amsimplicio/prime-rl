import verifiers as vf


def load_environment(
    **kwargs
):
    wordle = vf.load_environment("wordle")
    gsm8k = vf.load_environment("gsm8k")
    hendrycksmath = vf.load_environment("hendrycks-math")
    ifpt = vf.load_environment("vf-pt-instruction-following")

    return vf.EnvGroup(
        envs      = [ gsm8k , hendrycksmath  ],
        env_names = ["gsm8k","hendrycks-math"],
        **kwargs
    )
