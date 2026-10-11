#!/usr/bin/env python3
"""Build and run TRACE-FL in Docker.

    python run.py                 interactive menu
    python run.py sim             run the simulation
    python run.py deploy          run with server and clients as separate containers
    python run.py test [ARGS]     run pytest (ARGS passed through)
    python run.py shell           open a shell in the container
    python run.py rebuild         rebuild the image from scratch

Standard library only, so any Python 3 on the host can run it.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config" / "config.yaml"
SERVICE = "trace-fl"
IMAGE = "trace-fl:latest"

DEPLOY_PROJECT = "trace-fl-deploy"
# Rough estimates: each client container loads its own PyTorch + MNIST.
DEPLOY_BASE_GB = 1.5
DEPLOY_GB_PER_CLIENT = 0.6

EDITABLE = [
    ("num_clients", "num_clients", int, "positive integer"),
    ("num_rounds", "num_rounds", int, "positive integer"),
    ("name", "dataset", str, "mnist | fashion_mnist"),
    ("partition_type", "partition_type", str, "iid | dirichlet"),
    ("alpha", "alpha (Dirichlet)", float, "positive float (e.g. 1.0, 0.5, 0.1)"),
    ("epochs", "epochs", int, "positive integer"),
    ("batch_size", "batch_size", int, "positive integer"),
    ("learning_rate", "learning_rate", float, "positive float"),
    ("seed", "seed", int, "integer"),
]


def compose(*args):
    sys.stdout.flush()
    return subprocess.call(["docker", "compose", *args], cwd=str(ROOT))


def check_docker():
    if shutil.which("docker") is None:
        sys.exit("Docker is not installed. Install Docker Desktop (macOS/Windows) or Docker Engine (Linux).")
    if subprocess.call(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        sys.exit("Docker is installed but not running. Start Docker Desktop and try again.")


def run_sim(config_text=None, deploy=False):
    """Run with config_text (default: config/config.yaml) without modifying the tracked file."""
    if config_text is None:
        config_text = CONFIG.read_text()

    run_dir = ROOT / "experiments" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = run_dir / "config.yaml"
    run_config.write_text(config_text)
    print("Run config saved to {}".format(run_config.relative_to(ROOT)))

    if deploy:
        num_clients = int(read_settings(config_text)["num_clients"])
        return run_deployment(run_dir, run_config, num_clients)

    return compose(
        "run", "--rm", "--build",
        "-v", "{}:/app/config/config.yaml:ro".format(run_config),
        SERVICE,
    )


def docker_memory_gb():
    out = subprocess.run(
        ["docker", "info", "--format", "{{.MemTotal}}"], capture_output=True, text=True
    ).stdout.strip()
    return int(out) / 1e9 if out.isdigit() else None


def deployment_fits(num_clients):
    needed = DEPLOY_BASE_GB + DEPLOY_GB_PER_CLIENT * num_clients
    available = docker_memory_gb()
    if available is None:
        return True
    print("Deployment mode: {} client containers need ~{:.1f} GB; Docker has {:.1f} GB.".format(
        num_clients, needed, available))
    if needed <= available * 0.9:
        return True
    print("This may run out of memory. Lower num_clients or give Docker more memory.")
    return not sys.stdin.isatty() or ask_yes_no("Continue anyway?")


def deployment_spec(run_config, flwr_config, num_clients):
    services = {
        "superlink": {
            "image": IMAGE,
            "command": ["flower-superlink", "--insecure", "--host", "0.0.0.0"],
        },
        # Submits the run (`flwr run`) and polls node status; stays idle otherwise.
        "cli": {
            "image": IMAGE,
            "command": ["sleep", "infinity"],
            "init": True,
            "volumes": [
                {"type": "bind", "source": str(run_config),
                 "target": "/app/config/config.yaml", "read_only": True},
                {"type": "bind", "source": str(flwr_config),
                 "target": "/root/.flwr/config.toml"},
            ],
            "depends_on": ["superlink"],
        },
    }
    for i in range(num_clients):
        services["supernode-{}".format(i)] = {
            "image": IMAGE,
            "command": [
                "flower-supernode", "--insecure", "--superlink", "superlink:9092",
                "--node-config", "partition-id={} num-partitions={}".format(i, num_clients),
            ],
            "volumes": [
                {"type": "bind", "source": str(run_config),
                 "target": "/app/config/config.yaml", "read_only": True},
            ],
            "depends_on": ["superlink"],
        }
    return {"name": DEPLOY_PROJECT, "services": services}


def wait_for_supernodes(compose_file, num_clients, timeout=180):
    cmd = ["docker", "compose", "-f", str(compose_file), "exec", "-T", "cli",
           "flwr", "supernode", "ls", "deploy", "--format", "json"]
    deadline = time.monotonic() + timeout
    online = 0
    while time.monotonic() < deadline:
        out = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True).stdout
        online = len(re.findall(r'"status":\s*"online"', out))
        print("\rWaiting for clients: {}/{} online".format(online, num_clients), end="", flush=True)
        if online >= num_clients:
            print()
            return True
        time.sleep(2)
    print("\nOnly {}/{} clients came online within {}s.".format(online, num_clients, timeout))
    return False


def run_deployment(run_dir, run_config, num_clients):
    if not deployment_fits(num_clients):
        return 1

    flwr_config = run_dir / "flwr-config.toml"
    flwr_config.write_text(
        '[superlink]\ndefault = "deploy"\n\n'
        '[superlink.deploy]\naddress = "superlink:8000"\ninsecure = true\n'
    )
    compose_file = run_dir / "docker-compose.deploy.json"
    compose_file.write_text(json.dumps(deployment_spec(run_config, flwr_config, num_clients), indent=2))

    def deploy_compose(*args):
        return compose("-f", str(compose_file), *args)

    if compose("build") != 0:
        return 1
    try:
        if deploy_compose("up", "-d") != 0:
            return 1
        if not wait_for_supernodes(compose_file, num_clients):
            return 1
        return deploy_compose("exec", "cli", "flwr", "run", ".", "deploy", "--stream")
    finally:
        with open(str(run_dir / "containers.log"), "w") as log:
            subprocess.call(["docker", "compose", "-f", str(compose_file), "logs", "--no-color"],
                            cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        print("Container logs saved to {}".format((run_dir / "containers.log").relative_to(ROOT)))
        deploy_compose("down", "--remove-orphans")


def run_tests(pytest_args=()):
    return compose("run", "--rm", "--build", SERVICE, "pytest", *pytest_args)


def open_shell():
    return compose("run", "--rm", "--build", SERVICE, "bash")


def rebuild():
    return compose("build", "--no-cache")


def ask(prompt, default=""):
    answer = input(prompt).strip()
    return answer or default


def ask_yes_no(prompt, default=False):
    suffix = " [Y/n] " if default else " [y/N] "
    answer = ask(prompt + suffix).lower()
    return default if not answer else answer.startswith("y")


def _key_pattern(key):
    return re.compile(r"^(\s*" + re.escape(key) + r":\s*)(\S+)", re.MULTILINE)


def read_settings(text):
    settings = {}
    for key, _, _, _ in EDITABLE:
        match = _key_pattern(key).search(text)
        if match:
            settings[key] = match.group(2)
    return settings


def edit_config():
    """Prompt for settings; return the config text to run with, or None if unchanged."""
    text = CONFIG.read_text()
    current = read_settings(text)

    print("\nDefault settings (config/config.yaml):")
    for key, label, _, _ in EDITABLE:
        if key in current:
            val = current[key]
            if key == "alpha" and current.get("partition_type") == "iid":
                val += " (inactive for iid)"
            print("  {:<18} {}".format(label, val))

    if not ask_yes_no("\nChange any settings for this run?"):
        return None

    print("Press Enter to keep a value.")
    changed = False
    new_values = dict(current)
    for key, label, cast, hint in EDITABLE:
        if key not in current:
            continue
        # Skip prompting alpha if partition is iid
        if key == "alpha" and new_values.get("partition_type") == "iid":
            continue
        while True:
            value = ask("  {} [{}] ({}): ".format(label, current[key], hint), current[key]).strip()
            if key == "name":
                val_lower = value.lower()
                if val_lower in ("mnist", "fashion_mnist"):
                    value = val_lower
                    break
                print("    Invalid dataset. Choose 'mnist' or 'fashion_mnist'.")
            elif key == "partition_type":
                val_lower = value.lower()
                if val_lower in ("iid", "dirichlet"):
                    value = val_lower
                    break
                print("    Invalid partition type. Choose 'iid' or 'dirichlet'.")
            elif key == "alpha":
                try:
                    fval = float(value)
                    if fval <= 0:
                        raise ValueError
                    value = str(fval)
                    break
                except ValueError:
                    print("    Enter a positive float (e.g. 1.0, 0.5, 0.1).")
            else:
                try:
                    cval = cast(value)
                    if cval <= 0 and key != "seed":
                        raise ValueError
                    value = str(cval)
                    break
                except ValueError:
                    print("    Enter a positive {}.".format(cast.__name__))
        new_values[key] = value
        if value != current[key]:
            text = _key_pattern(key).sub(lambda m: m.group(1) + value, text, count=1)
            changed = True

    return text if changed else None


def interactive_sim():
    override = edit_config()
    deploy = ask_yes_no(
        "\nRun server and clients as separate containers (deployment mode, more memory)?")
    code = run_sim(override, deploy=deploy)
    if override is not None and ask_yes_no("\nSave these settings as the new defaults in config/config.yaml?"):
        CONFIG.write_text(override)
        print("Saved config/config.yaml.")
    return code


def parse_sim_args(args):
    """Parse CLI flags to override configuration for a simulation run."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="python run.py sim",
        description="Run TRACE-FL simulation in Docker with custom dataset split and training options."
    )
    parser.add_argument("--dataset", "-d", choices=["mnist", "fashion_mnist"], dest="name",
                        help="Dataset: 'mnist' or 'fashion_mnist'")
    parser.add_argument("--partition", "-p", choices=["iid", "dirichlet"], dest="partition_type",
                        help="Partition type: 'iid' or 'dirichlet'")
    parser.add_argument("--alpha", "-a", type=float,
                        help="Dirichlet concentration alpha (e.g. 1.0, 0.5, 0.1)")
    parser.add_argument("--clients", "-c", type=int, dest="num_clients",
                        help="Number of federated clients")
    parser.add_argument("--rounds", "-r", type=int, dest="num_rounds",
                        help="Number of FL rounds")
    parser.add_argument("--epochs", "-e", type=int,
                        help="Local epochs per round")
    parser.add_argument("--batch-size", "-b", type=int, dest="batch_size",
                        help="Batch size")
    parser.add_argument("--lr", type=float, dest="learning_rate",
                        help="Learning rate")
    parser.add_argument("--seed", "-s", type=int,
                        help="Random seed")
    parser.add_argument("--deploy", action="store_true",
                        help="Run in deployment mode (multi-container)")

    # Member 3: Attack Engine CLI flags
    parser.add_argument("--attack", dest="attack_enabled", action="store_true", default=None,
                        help="Enable attack engine")
    parser.add_argument("--no-attack", dest="attack_enabled", action="store_false",
                        help="Disable attack engine")
    parser.add_argument("--attack-type", choices=["label_flipping", "gaussian_noise", "sign_flip", "scaling"],
                        help="Attack type to inject")
    parser.add_argument("--attack-intensity", type=float,
                        help="Attack intensity (noise std, scale multiplier, flip ratio)")
    parser.add_argument("--attack-mode", choices=["continuous", "intermittent", "custom_rounds", "probabilistic"],
                        help="Attack schedule mode")
    parser.add_argument("--attack-start", type=int,
                        help="Round number to start attack")
    parser.add_argument("--attack-end", type=int,
                        help="Round number to end attack")
    parser.add_argument("--attack-interval", type=int,
                        help="Attack interval for intermittent attacks")
    parser.add_argument("--attack-clients", type=str,
                        help="Target client IDs (e.g. '0' or '0,1') or fraction (e.g. '0.2')")
    parser.add_argument("--aggregation", choices=["fedavg", "median", "trimmed_mean"],
                        help="Server aggregation rule")
    parser.add_argument("--max-update-norm", type=float, dest="max_update_norm",
                        help="Reject client updates with ||w - w_global|| above this (norm bounding)")
    return parser.parse_known_args(args)


def run_sim_cli(args):
    parsed, unknown = parse_sim_args(args)
    text = CONFIG.read_text()
    try:
        import yaml
        cfg = yaml.safe_load(text)
        changed = False
        if parsed.name is not None:
            cfg.setdefault("dataset", {})["name"] = parsed.name
            changed = True
        if parsed.partition_type is not None:
            cfg.setdefault("dataset", {})["partition_type"] = parsed.partition_type
            changed = True
        if parsed.alpha is not None:
            if parsed.alpha <= 0:
                sys.exit("Error: --alpha must be positive.")
            cfg.setdefault("dataset", {})["alpha"] = parsed.alpha
            changed = True
        if parsed.num_clients is not None:
            if parsed.num_clients <= 0:
                sys.exit("Error: --clients must be positive.")
            cfg["num_clients"] = parsed.num_clients
            changed = True
        if parsed.num_rounds is not None:
            if parsed.num_rounds <= 0:
                sys.exit("Error: --rounds must be positive.")
            cfg["num_rounds"] = parsed.num_rounds
            changed = True
        if parsed.epochs is not None:
            if parsed.epochs <= 0:
                sys.exit("Error: --epochs must be positive.")
            cfg.setdefault("local_training", {})["epochs"] = parsed.epochs
            changed = True
        if parsed.batch_size is not None:
            if parsed.batch_size <= 0:
                sys.exit("Error: --batch-size must be positive.")
            cfg.setdefault("local_training", {})["batch_size"] = parsed.batch_size
            changed = True
        if parsed.learning_rate is not None:
            if parsed.learning_rate <= 0:
                sys.exit("Error: --lr must be positive.")
            cfg.setdefault("local_training", {})["learning_rate"] = parsed.learning_rate
            changed = True
        if parsed.seed is not None:
            cfg["seed"] = parsed.seed
            changed = True

        # Attack configuration overrides
        if parsed.attack_enabled is not None:
            cfg.setdefault("attack", {})["enabled"] = parsed.attack_enabled
            changed = True
        if parsed.attack_type is not None:
            cfg.setdefault("attack", {})["type"] = parsed.attack_type
            cfg["attack"]["enabled"] = True
            changed = True
        if parsed.attack_intensity is not None:
            cfg.setdefault("attack", {})["intensity"] = parsed.attack_intensity
            changed = True
        if parsed.attack_mode is not None:
            cfg.setdefault("attack", {}).setdefault("schedule", {})["mode"] = parsed.attack_mode
            changed = True
        if parsed.attack_start is not None:
            cfg.setdefault("attack", {}).setdefault("schedule", {})["start_round"] = parsed.attack_start
            changed = True
        if parsed.attack_end is not None:
            cfg.setdefault("attack", {}).setdefault("schedule", {})["end_round"] = parsed.attack_end
            changed = True
        if parsed.attack_interval is not None:
            cfg.setdefault("attack", {}).setdefault("schedule", {})["interval"] = parsed.attack_interval
            changed = True
        if parsed.attack_clients is not None:
            cfg.setdefault("attack", {}).setdefault("targets", {})
            raw = parsed.attack_clients.strip()
            if "." in raw and not ("," in raw):
                try:
                    frac = float(raw)
                    cfg["attack"]["targets"]["mode"] = "fraction"
                    cfg["attack"]["targets"]["fraction"] = frac
                except ValueError:
                    cfg["attack"]["targets"]["mode"] = "client_ids"
                    cfg["attack"]["targets"]["client_ids"] = [int(c.strip()) for c in raw.split(",")]
            else:
                cfg["attack"]["targets"]["mode"] = "client_ids"
                cfg["attack"]["targets"]["client_ids"] = [int(c.strip()) for c in raw.split(",") if c.strip()]
            changed = True

        if parsed.aggregation is not None:
            cfg.setdefault("federation", {})["aggregation"] = parsed.aggregation
            changed = True
        if parsed.max_update_norm is not None:
            cfg.setdefault("federation", {})["max_update_norm"] = parsed.max_update_norm
            changed = True

        if changed:
            text = yaml.dump(cfg, sort_keys=False)
            return run_sim(text, deploy=parsed.deploy)
        return run_sim(None, deploy=parsed.deploy)
    except ImportError:
        # Without PyYAML only flat keys can be rewritten. Attack/aggregation flags
        # are nested, and used to be dropped silently (an "attack" run that was honest).
        nested = [f for f in ("attack_enabled", "attack_type", "attack_intensity", "attack_mode",
                              "attack_start", "attack_end", "attack_interval", "attack_clients",
                              "aggregation", "max_update_norm")
                  if getattr(parsed, f) is not None]
        if nested:
            sys.exit("Error: --{} need PyYAML on the host. Run `pip install pyyaml` "
                     "or `uv run python run.py ...`.".format(", --".join(f.replace("_", "-") for f in nested)))
        changed = False
        for key, _, _, _ in EDITABLE:
            val = getattr(parsed, key, None)
            if val is not None:
                if key == "alpha" and val <= 0:
                    sys.exit("Error: --alpha must be positive.")
                if key in ("num_clients", "num_rounds", "epochs", "batch_size", "learning_rate") and val <= 0:
                    sys.exit("Error: --{} must be positive.".format(key))
                text = _key_pattern(key).sub(lambda m: m.group(1) + str(val), text, count=1)
                changed = True
        return run_sim(text if changed else None, deploy=parsed.deploy)


def menu():
    options = {
        "1": ("Run simulation", interactive_sim),
        "2": ("Run tests", lambda: run_tests(ask("Test file or pytest args (blank = all): ").split())),
        "3": ("Open a shell in the container", open_shell),
        "4": ("Rebuild image from scratch (only if something seems broken)", rebuild),
    }
    print("TRACE-FL (Docker)")
    for key, (label, _) in options.items():
        print("  {}) {}".format(key, label))
    print("  q) Quit")

    choice = ask("\nChoose: ").lower()
    if choice not in options:
        return 0
    return options[choice][1]()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sim":
        check_docker()
        return run_sim_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "deploy":
        check_docker()
        return run_sim_cli(sys.argv[2:] + ["--deploy"])
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        check_docker()
        return run_tests(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "shell":
        check_docker()
        return open_shell()
    elif len(sys.argv) > 1 and sys.argv[1] == "rebuild":
        check_docker()
        return rebuild()
    elif len(sys.argv) > 1:
        sys.exit(__doc__)

    check_docker()
    return menu()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print()
        sys.exit(130)
