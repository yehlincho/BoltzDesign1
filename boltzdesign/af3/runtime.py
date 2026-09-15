"""Load AlphaFold 3 runtime symbols from the user's own AlphaFold 3 installation.

AlphaFold 3 source is licensed CC BY-NC-SA 4.0 and its weights require agreeing to
DeepMind's terms, so it is never redistributed here. Install it yourself (see
https://github.com/google-deepmind/alphafold3) and point this module at it via
--alphafold_dir, $AF3_ROOT, or the default ~/alphafold3.

The symbols below come from that installation's run_alphafold.py, except Input,
which is the public alphafold3.common.folding_input.Input.
"""

import importlib.util
import os
import sys

_CACHE = {}


def af3_root(path=None):
    """Resolve the AlphaFold 3 install directory."""
    return os.path.expanduser(path or os.environ.get("AF3_ROOT", "~/alphafold3"))


def load_runtime(path=None):
    """Import run_alphafold.py from the AF3 install and return the module."""
    root = af3_root(path)
    if root in _CACHE:
        return _CACHE[root]

    script = os.path.join(root, "run_alphafold.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(
            f"AlphaFold 3 not found at {root!r} (expected {script}). Install AlphaFold 3 from "
            "https://github.com/google-deepmind/alphafold3 and set --alphafold_dir or $AF3_ROOT."
        )

    # Append (never prepend): the AF3 checkout contains an uncompiled alphafold3/ source
    # tree that would otherwise shadow the installed alphafold3 package.
    if root not in sys.path:
        sys.path.append(root)

    spec = importlib.util.spec_from_file_location("af3_run_alphafold", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["af3_run_alphafold"] = module
    spec.loader.exec_module(module)
    _CACHE[root] = module
    return module


def get_symbols(path=None):
    """Return the AF3 entry points the validator needs.

    (make_model_config, ModelRunner, predict_structure, write_outputs, Input)

    Normally these come from run_alphafold.py in the AF3 install. If $AF3_PIPELINE
    points at an equivalent module, that is used instead -- needed when the local AF3
    is customised and its stock run_alphafold.py no longer matches the installed package.
    """
    override = os.environ.get("AF3_PIPELINE")
    if override:
        spec = importlib.util.spec_from_file_location("af3_pipeline_override", os.path.expanduser(override))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["af3_pipeline_override"] = mod
        spec.loader.exec_module(mod)
        return (mod.make_model_config, mod.ModelRunner, mod.predict_structure, mod.write_outputs, mod.Input)

    mod = load_runtime(path)
    from alphafold3.common import folding_input

    missing = [
        n for n in ("make_model_config", "ModelRunner", "predict_structure", "write_outputs")
        if not hasattr(mod, n)
    ]
    if missing:
        raise ImportError(
            f"AlphaFold 3 run_alphafold.py at {af3_root(path)!r} is missing {missing}. "
            "An AlphaFold 3 release providing these entry points is required."
        )

    return (
        mod.make_model_config,
        mod.ModelRunner,
        mod.predict_structure,
        mod.write_outputs,
        folding_input.Input,
    )
