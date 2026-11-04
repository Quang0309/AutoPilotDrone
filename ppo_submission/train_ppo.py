"""Fresh PPO curriculum; each stage trains for its target and until 100/100 gates pass."""
import os
from pathlib import Path
import sys

SUBMISSION_DIR = Path(__file__).resolve().parent
ROOT = SUBMISSION_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'True')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(SUBMISSION_DIR / '.cache/matplotlib'))
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import random
import shutil
from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecFrameStack
from ppo_submission.runtime import NavigateAviary, NavigateExtractor
from gym_pybullet_drones.utils.enums import ActionType, ObservationType

STAGES = ['easy', 'medium', 'hard']

def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def checkpoint(model, path):
    temporary = path.with_name(path.stem + '_pending.zip')
    model.save(temporary)
    temporary.replace(path)

def make_env(stage, count=1, seed=0, monitor=None):
    env = make_vec_env(NavigateAviary, n_envs=count, seed=seed, monitor_dir=monitor,
                       env_kwargs=dict(difficulty=stage, gui=False,
                                       obs=ObservationType.ALL, act=ActionType.VEL))
    return VecFrameStack(env, 4, channels_order='first')

def evaluate(model, env, seeds):
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    rows = []
    try:
        for seed in seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            env.seed(seed)
            observation = env.reset()
            total, length, collision = 0., 0, False
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, rewards, done, infos = env.step(action)
                total += float(rewards[0]); length += 1
                collision |= bool(infos[0]['collision'])
                if done[0]:
                    info = infos[0]
                    rows.append(dict(seed=seed, reward=total, length=length,
                                     success=bool(info['success']) and not collision,
                                     collision=collision, distance=info['distance'],
                                     reason=info['truncation_reason']))
                    break
    finally:
        random.setstate(states[0]); np.random.set_state(states[1]); torch.set_rng_state(states[2])
    return dict(episodes=len(rows), success_rate=float(np.mean([r['success'] for r in rows])),
                collision_rate=float(np.mean([r['collision'] for r in rows])),
                mean_reward=float(np.mean([r['reward'] for r in rows])),
                mean_length=float(np.mean([r['length'] for r in rows])), results=rows)

def passed(result):
    return result['episodes'] == 100 and len(result['results']) == 100 and all(
        row['success'] and not row['collision'] for row in result['results'])

class History(BaseCallback):
    def __init__(self, directory, env, origin, target, interval):
        super().__init__()
        self.directory, self.env, self.origin = directory, env, origin
        self.target, self.interval = target, interval
        self.gate, self.validation, self.best = False, None, None
        self.next_eval, self.next_save = interval, 5000
        self.attempts = len(list(directory.glob('validation_attempt_*.json')))
        metrics = directory/'best_metrics.json'
        if metrics.exists():
            row=json.loads(metrics.read_text())
            self.best = row['success_rate'], -row['collision_rate'], row['mean_reward']

    def record(self):
        result = evaluate(self.model, self.env, range(200000,200100))
        steps = self.model.num_timesteps-self.origin
        result.update(timesteps=self.model.num_timesteps, stage_steps=steps,
                      evaluated_at=datetime.now().astimezone().isoformat())
        score = result['success_rate'], -result['collision_rate'], result['mean_reward']
        if self.best is None or score > self.best:
            self.best = score
            checkpoint(self.model,self.directory/'best_model.zip')
            save_json(self.directory/'best_metrics.json',result)
        checkpoint(self.model,self.directory/'latest_model.zip')
        if steps >= self.target and passed(result):
            self.attempts += 1
            start = 1000000+self.attempts*1000
            self.validation = evaluate(self.model,self.env,range(start,start+100))
            save_json(self.directory/f'validation_attempt_{self.attempts:04d}.json',self.validation)
            self.gate = passed(self.validation)
            if self.gate:
                # The exported best model must be the policy verified for transfer.
                checkpoint(self.model,self.directory/'best_model.zip')
                checkpoint(self.model,self.directory/'passed_model.zip')
                save_json(self.directory/'best_metrics.json',dict(result,gate_passed=True))
                save_json(self.directory/'passed_metrics.json',result)
        result['gate_passed'] = self.gate
        with (self.directory/'evaluations.jsonl').open('a') as handle:
            handle.write(json.dumps(result)+'\n')
        save_json(self.directory/'status.json',dict(state='training',**{k:v for k,v in result.items() if k!='results'}))
        print('[EVAL]',json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
        self.refresh_charts()

    def refresh_charts(self):
        from ppo_submission.plots import refresh
        refresh(self.directory.parent)

    def _on_step(self):
        steps = self.num_timesteps-self.origin
        if steps >= self.next_save:
            checkpoint(self.model,self.directory/'latest_model.zip')
            save_json(self.directory/'progress_status.json',dict(timesteps=self.num_timesteps,
                      stage_steps=steps,updated_at=datetime.now().astimezone().isoformat()))
            self.next_save = steps+5000
        if steps >= self.next_eval:
            self.record()
            self.next_eval += self.interval
        return not self.gate

def run(args):
    import gym_pybullet_drones
    assert Path(gym_pybullet_drones.__file__).resolve().parent == ROOT/'gym_pybullet_drones'
    torch.set_num_threads(1)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    output = Path(args.output).resolve()
    recorded = SUBMISSION_DIR/'ppo_results'
    if output == recorded or recorded in output.parents:
        raise ValueError('Use a new output directory to preserve the recorded run.')
    if not args.resume and any((output/stage).exists() for stage in STAGES):
        raise FileExistsError('Stage data exists; use a new output directory or --resume.')
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    save_json(output/('resume_config.json' if args.resume else 'run_config.json'),dict(
        vars(args),repository=str(ROOT),pid=os.getpid(),started_at=datetime.now().astimezone().isoformat(),
        reference='example_training_mimic_uts.ipynb',device='cpu',seed=42,
        transition='minimum target steps + 100/100 screening + independent 100/100 validation'))
    save_json(output/'versions.json',{name:importlib.metadata.version(name) for name in
        ['torch','stable-baselines3','gymnasium','numpy','pybullet']})
    snapshot=output/'source'/datetime.now().strftime('%Y%m%d_%H%M%S');snapshot.mkdir(parents=True)
    sources = [
        ROOT/'gym_pybullet_drones/envs/NavigateAviary.py',
        ROOT/'gym_pybullet_drones/control/DSLPIDControl.py',
        SUBMISSION_DIR/'runtime.py',
        SUBMISSION_DIR/'train_ppo.py',
    ]
    for source in sources:
        shutil.copy2(source,snapshot/source.name)
    shutil.copy2(ROOT/'example_training_mimic_uts.ipynb',snapshot/'reference.ipynb')
    previous = None
    for stage,target,lr in zip(STAGES,args.steps,[1e-4,1e-4,1e-5]):
        directory=output/stage
        status_path=directory/'status.json'
        prior=json.loads(status_path.read_text()) if status_path.exists() else {}
        if args.resume and prior.get('state')=='complete':
            assert passed(json.loads((directory/'validation.json').read_text()))
            previous=directory/'best_model.zip'
            continue
        directory.mkdir(exist_ok=args.resume)
        resuming=args.resume and (directory/'latest_model.zip').exists()
        logs=directory/'segments'/datetime.now().strftime('%Y%m%d_%H%M%S') if resuming else directory
        train_env=make_env(stage,args.n_envs,monitor=str(logs/'monitor'))
        eval_env=make_env(stage,seed=200000)
        model=PPO('MultiInputPolicy',train_env,device='cpu',learning_rate=lr,n_steps=2048,batch_size=1024,
                  n_epochs=10,clip_range=.15,target_kl=.015,ent_coef=.005,vf_coef=.05,verbose=0,
                  policy_kwargs=dict(features_extractor_class=NavigateExtractor,
                    features_extractor_kwargs=dict(cnn_output_dim=32),share_features_extractor=False,net_arch=[64,64]))
        origin=0
        if resuming:
            origin=json.loads((directory/'run_config.json').read_text())['source_timesteps']
            candidates=[directory/'latest_model.zip',directory/'interrupted_model.zip']
            resume_path=max((path for path in candidates if path.exists()),key=lambda path:path.stat().st_mtime_ns)
            model=PPO.load(resume_path,env=train_env,device='cpu')
        elif previous:
            assert previous.resolve().is_relative_to(output)
            old=PPO.load(previous,device='cpu')
            model.policy.load_state_dict(old.policy.state_dict(),strict=True)
            model.policy.log_std.data.fill_(-1.2)
            model.num_timesteps=origin=old.num_timesteps
        if not resuming:
            save_json(directory/'run_config.json',dict(stage=stage,target_steps=target,source_timesteps=origin,
                source_model=str(previous) if previous else None,
                source_sha256=hashlib.sha256(previous.read_bytes()).hexdigest() if previous else None,
                initialization='previous passed stage' if previous else 'random weights',learning_rate=lr))
            checkpoint(model,directory/'initial_model.zip')
        model.set_logger(configure(str(logs),['csv','tensorboard']))
        save_json(output/'status.json',dict(state='training',stage=stage,pid=os.getpid()))
        callback=History(directory,eval_env,origin,target,args.eval_interval)
        callback.init_callback(model)
        callback.next_eval=((model.num_timesteps-origin)//args.eval_interval+1)*args.eval_interval
        callback.next_save=model.num_timesteps-origin+5000
        try:
            callback.record()
            while not callback.gate:
                model.learn(total_timesteps=max(target,16384),reset_num_timesteps=False,callback=callback,log_interval=1)
                if not callback.gate:callback.record()
            checkpoint(model,directory/'final_model.zip')
            save_json(directory/'validation.json',callback.validation)
            save_json(directory/'status.json',dict(state='complete',gate_passed=True,
                stage_steps=model.num_timesteps-origin,validation={k:v for k,v in callback.validation.items() if k!='results'}))
        except BaseException:
            checkpoint(model,directory/'interrupted_model.zip')
            save_json(directory/'status.json',dict(state='interrupted',timesteps=model.num_timesteps))
            save_json(output/'status.json',dict(state='interrupted',stage=stage))
            raise
        finally:
            train_env.close();eval_env.close()
        previous=directory/'best_model.zip'
    save_json(output/'status.json',dict(state='complete',stages=STAGES))

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',default=str(SUBMISSION_DIR/'outputs/ppo_curriculum'))
    parser.add_argument('--steps',nargs=3,type=int,default=[500000,400000,300000])
    parser.add_argument('--n-envs',type=int,default=8)
    parser.add_argument('--eval-interval',type=int,default=20000)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.steps+[args.n_envs,args.eval_interval])<=0:parser.error('Sizes must be positive.')
    run(args)
