#!/usr/bin/env python3
import numpy as np
import json, os, time
from collections import deque
from datetime import datetime


class TinyNN:
    def __init__(self, in_dim, hidden, out_dim, lr=0.01):
        self.W1 = np.random.randn(in_dim, hidden).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, out_dim).astype(np.float32) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros(out_dim, dtype=np.float32)
        self.lr = lr
        self.cache = {}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        self.cache = {"x": x, "z1": z1, "a1": a1, "z2": z2}
        return z2

    def backward(self, grad_out):
        x = self.cache["x"]
        a1 = self.cache["a1"]
        z1 = self.cache["z1"]
        grad_W2 = a1.T @ grad_out
        grad_b2 = grad_out.sum(axis=0)
        grad_a1 = grad_out @ self.W2.T
        grad_z1 = grad_a1 * (z1 > 0)
        grad_W1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2


class WorldModel:
    def __init__(self, state_dim, latent_dim, lr=0.005):
        self.net = TinyNN(state_dim, latent_dim, state_dim + 1, lr)

    def predict(self, state):
        out = self.net.forward(state)
        return out[:, :-1], out[:, -1:]

    def train_step(self, state, next_state, reward):
        pred = self.net.forward(state)
        target = np.concatenate([next_state, reward], axis=-1)
        grad = (pred - target) / state.shape[0]
        self.net.backward(grad)
        return float(np.mean((pred - target) ** 2))


class Expert:
    def __init__(self, in_dim, hidden, out_dim, lr=0.01):
        self.net = TinyNN(in_dim, hidden, out_dim, lr)

    def forward(self, x):
        return self.net.forward(x)

    def backward(self, grad):
        self.net.backward(grad)


class Amygdala:
    def __init__(self):
        self.w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.b = 0.0
        self.lr = 0.01

    def forward(self, reward, uncertainty, novelty):
        s = (self.w[0] * reward + self.w[1] * uncertainty +
             self.w[2] * novelty + self.b)
        return 1.0 / (1.0 + np.exp(-s))


class BasalGanglia:
    def __init__(self, n_experts):
        self.n = n_experts
        self.W = np.random.randn(n_experts, n_experts).astype(np.float32) * 0.1

    def forward(self, x):
        logits = x @ self.W
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)


class Replay:
    def __init__(self, capacity=500):
        self.buf = deque(maxlen=capacity)

    def push(self, exp):
        self.buf.append(exp)

    def sample(self, n):
        if len(self.buf) < n:
            return list(self.buf)
        idx = np.random.choice(len(self.buf), n, replace=False)
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


class SelfEvaluator:
    def __init__(self):
        self.history = deque(maxlen=200)

    def evaluate(self, reward, uncertainty):
        score = float(reward) - 0.3 * float(uncertainty)
        self.history.append(score)
        return {"score": score, "trend": self._trend(),
                "avg": float(np.mean(self.history)) if self.history else 0.0}

    def _trend(self):
        if len(self.history) < 20:
            return "unknown"
        recent = np.mean(list(self.history)[-10:])
        older = np.mean(list(self.history)[-20:-10])
        d = recent - older
        if d > 0.05: return "improving"
        if d < -0.05: return "declining"
        return "stable"


class MetaLearner:
    def __init__(self, agent):
        self.agent = agent
        self.history = []

    def step(self, eval_result):
        self.history.append(eval_result["score"])
        if len(self.history) < 30:
            return "collecting"
        recent = np.mean(self.history[-15:])
        older = np.mean(self.history[-30:-15])
        if recent < older - 0.1:
            for e in self.agent.experts:
                e.net.lr = max(0.001, e.net.lr * 0.95)
            return "lr_down"
        elif recent > older + 0.1:
            for e in self.agent.experts:
                e.net.lr = min(0.05, e.net.lr * 1.02)
            return "lr_up"
        return "stable"


class UNCALite:
    def __init__(self, state_dim=8, latent_dim=16, n_experts=3):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.n_experts = n_experts
        self.world = WorldModel(state_dim, latent_dim)
        self.experts = [Expert(state_dim + 1, 32, 1) for _ in range(n_experts)]
        self.amygdala = Amygdala()
        self.gate = BasalGanglia(n_experts)
        self.replay = Replay(capacity=500)
        self.evaluator = SelfEvaluator()
        self.meta = MetaLearner(self)
        self.cycle = 0
        self.birth = datetime.now().isoformat()

    def forward(self, state_seq, action):
        z, r_pred = self.world.predict(state_seq)
        z = z[:, :self.latent_dim]
        x = np.concatenate([z, action], axis=-1)
        preds = np.stack([e.forward(x) for e in self.experts], axis=-1)
        novelty = preds.std(axis=-1, keepdims=True).mean()
        uncertainty = preds.std(axis=-1).mean()
        salience = self.amygdala.forward(
            float(r_pred.mean()), float(uncertainty), float(novelty))
        gate_input = np.array([[salience] * self.n_experts], dtype=np.float32)
        probs = self.gate.forward(gate_input)
        chosen = int(np.argmax(probs[0]))
        chosen_pred = preds[:, :, chosen]
        return {"latent": z, "reward": r_pred, "preds": preds,
                "probs": probs, "chosen_idx": chosen,
                "chosen_pred": chosen_pred, "uncertainty": uncertainty,
                "novelty": novelty, "salience": salience}

    def step(self, state_seq, action, reward, next_state):
        out = self.forward(state_seq, action)
        self.replay.push({"state_seq": state_seq.copy(),
                          "action": action.copy(),
                          "reward": reward,
                          "next_state": next_state.copy()})
        loss_world = self.world.train_step(
            state_seq, next_state, np.array([[reward]]))
        if len(self.replay) >= 8:
            batch = self.replay.sample(8)
            for exp in batch:
                z, _ = self.world.predict(exp["state_seq"])
                x = np.concatenate([z, exp["action"]], axis=-1)
                for e in self.experts:
                    pred = e.forward(x)
                    err = (pred - exp["reward"]) / len(batch)
                    e.backward(err)
        eval_result = self.evaluator.evaluate(reward, out["uncertainty"])
        meta_action = self.meta.step(eval_result)
        self.cycle += 1
        return {"out": out, "loss_world": loss_world,
                "eval": eval_result, "meta": meta_action,
                "cycle": self.cycle}

    def save(self, path="~/unca_lite.npz"):
        path = os.path.expanduser(path)
        data = {"cycle": self.cycle, "birth": self.birth,
                "state_dim": self.state_dim, "latent_dim": self.latent_dim,
                "n_experts": self.n_experts,
                "amygdala_w": self.amygdala.w, "gate_W": self.gate.W,
                "eval_history": np.array(list(self.evaluator.history), dtype=np.float32),
                "meta_history": np.array(self.meta.history, dtype=np.float32),
                "world_W1": self.world.net.W1, "world_b1": self.world.net.b1,
                "world_W2": self.world.net.W2, "world_b2": self.world.net.b2}
        for i, e in enumerate(self.experts):
            data[f"exp{i}_W1"] = e.net.W1
            data[f"exp{i}_b1"] = e.net.b1
            data[f"exp{i}_W2"] = e.net.W2
            data[f"exp{i}_b2"] = e.net.b2
        np.savez(path, **data)
        return path

    def load(self, path="~/unca_lite.npz"):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return False
        try:
            d = np.load(path, allow_pickle=True)
            self.cycle = int(d["cycle"])
            self.birth = str(d["birth"])
            self.amygdala.w = d["amygdala_w"]
            self.gate.W = d["gate_W"]
            self.evaluator.history = deque(list(d["eval_history"]), maxlen=200)
            self.meta.history = list(d["meta_history"])
            self.world.net.W1 = d["world_W1"]
            self.world.net.b1 = d["world_b1"]
            self.world.net.W2 = d["world_W2"]
            self.world.net.b2 = d["world_b2"]
            for i, e in enumerate(self.experts):
                e.net.W1 = d[f"exp{i}_W1"]
                e.net.b1 = d[f"exp{i}_b1"]
                e.net.W2 = d[f"exp{i}_W2"]
                e.net.b2 = d[f"exp{i}_b2"]
            return True
        except Exception as ex:
            print(f"Load failed: {ex}")
            return False


def main():
    print("=" * 60)
    print("UNCA-Lite - Foundation Anan Mahdi")
    print("=" * 60)
    agent = UNCALite(state_dim=8, latent_dim=16, n_experts=3)
    if agent.load():
        print(f"Loaded state (cycle {agent.cycle})")
    else:
        print("Fresh start")
    n_cycles = int(input("Cycles (e.g. 50): ").strip() or "50")
    t0 = time.time()
    for i in range(n_cycles):
        state_seq = np.random.randn(1, agent.state_dim).astype(np.float32)
        next_state = state_seq + np.random.randn(1, agent.state_dim).astype(np.float32) * 0.1
        action = np.random.randn(1, 1).astype(np.float32)
        reward = 1.0 / (1.0 + np.mean((next_state - state_seq) ** 2))
        result = agent.step(state_seq, action, reward, next_state)
        if i % 10 == 0 or i == n_cycles - 1:
            ev = result["eval"]
            print(f"  cycle {i:4d} | score {ev['score']:+.3f} | "
                  f"{ev['trend']:10s} | loss {result['loss_world']:.4f} | "
                  f"{result['meta']}")
    dt = time.time() - t0
    print(f"\nTime: {dt:.1f}s ({n_cycles/dt:.1f} cycles/s)")
    path = agent.save()
    print(f"Saved: {path}")
    print(f"Total cycles: {agent.cycle}")


if __name__ == "__main__":
    main()
