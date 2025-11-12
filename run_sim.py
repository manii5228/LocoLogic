''' 
usage :

python run_sim.py --trains 50 --sim-hours 3 --dispatcher fcfs --seed 42 --out ./my_run1 --plot --animate --animate-blocks
python run_sim.py --trains 80 --sim-hours 4 --dispatcher priority --seed 123 --out ./my_run2 --plot --animate-blocks

'''
#!/usr/bin/env python3
import argparse
from pathlib import Path
import time as pytime

from rail.network import build_tn_network, generate_trains_network
from rail.simulator import NetworkSimulator
from rail.plotter import plot_network, plot_time_distance, animate_time_distance
from rail.visualizer import plot_trains_with_conflicts_timeline
from rail.animator import animate_trains_blocks  # new block-level animator

def main():
    parser = argparse.ArgumentParser(description="TN simulator (modular)")
    parser.add_argument("--trains", type=int, default=80)
    parser.add_argument("--sim-hours", type=float, default=4.0)
    parser.add_argument("--out", type=str, default="./out_full")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dispatcher", type=str, choices=["fcfs","priority","throughput"], default="priority")
    parser.add_argument("--plot", action="store_true", help="Generate station-level plots")
    parser.add_argument("--animate", action="store_true", help="Generate station-level animation")
    parser.add_argument("--animate-blocks", action="store_true", help="Generate block-level animation with conflicts")
    args = parser.parse_args()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Building network...")
    nodes, edges = build_tn_network()

    print(f"Generating {args.trains} trains...")
    trains = generate_trains_network(nodes, edges, args.trains, seed=args.seed)

    # Initialize simulator
    sim = NetworkSimulator(
        nodes, edges, trains,
        sim_hours=args.sim_hours,
        min_headway_s=60,
        dispatcher=args.dispatcher,
        seed=args.seed
    )
    sim.init()

    # Run simulation
    t0 = pytime.time()
    print("Running simulation...")
    sim.run()
    dt = pytime.time() - t0
    print(f"Simulation finished in {dt:.2f} s, timeline rows: {len(sim.timeline)}")

    # Save configuration and events
    sim.save_config(outdir)
    sim.save_events(outdir)

    # Plot network and timeline
    if args.plot:
        plot_network(nodes, edges, outdir)
        plot_time_distance(sim.timeline, nodes, outdir)
        plot_trains_with_conflicts_timeline(nodes, edges, sim.timeline)

    # Animate train movements (station-level)
    if args.animate:
        animate_time_distance(sim.timeline, nodes, outdir)

    # Animate train movements (block-level)
    if args.animate_blocks:
        animate_trains_blocks(nodes, edges, trains, max_time=args.sim_hours*60)

    # Print sample timeline
    print("\nSample timeline (first 30 rows):")
    for r in sorted(sim.timeline, key=lambda x: (x["time_s"], x.get("train","")))[:30]:
        print(r)

    # Compute throughput KPI
    finished = 0
    max_time_min = args.sim_hours * 60
    for train in trains:
        # estimate arrival at last edge using timeline
        arr_time = 0
        for ev in sim.timeline:
            if ev.get("train") == train.id:
                arr_time = max(arr_time, ev["time_s"]/60)
        if arr_time <= max_time_min:
            finished += 1
    total = len(trains)
    print(f"\nThroughput KPI: {finished}/{total} trains reached destination within {max_time_min:.0f} minutes")

    # Energy summary
    print("\nEnergy summary (top 10 trains):")
    energy_list = [(tid, st.get("energy_j",0.0)) for tid, st in sim.state.items()]
    energy_list.sort(key=lambda x: -x[1])
    for tid, ej in energy_list[:10]:
        print(f"  {tid}: {ej/1e6:.2f} MJ")


if __name__ == "__main__":
    main()
