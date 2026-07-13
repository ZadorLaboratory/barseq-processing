#!/usr/bin/env python3
import subprocess, shutil, os, sys, yaml, pathlib

ENVS=["barseq","n2v_tf24_gpu","cellpose_v3"]
OUT_DIR_YML=pathlib.Path("envs_export")
OUT_DIR_YML.mkdir(parents=True, exist_ok=True)


def _dump_yaml(env,args,outpath):
    raw=subprocess.check_output(["conda","env","export","-n",env,*args])
    data=yaml.safe_load(raw)
    data.pop("prefix",None)
    outpath.write_text(yaml.safe_dump(data,sort_keys=False))
    print(f"Wrote {outpath}")

def export_env(env):

    _dump_yaml(env,["--no-builds"],OUT_DIR_YML/f"{env}.pinned.yml")
    
def main():
    for env in ENVS:
        try:
            subprocess.check_call(["conda","run","-n",env,"python","-V"],
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[skip] env not found:{env}",file=sys.stderr); continue
        export_env(env)

if __name__=="__main__":
    main()
