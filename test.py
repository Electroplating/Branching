from omt_branching.solver.decide_omt import solve_omt_with_decider, smt2_to_instance
from omt_branching.solver.instance_gen import OMTInstance
inst = smt2_to_instance("examples/artifacts/decide_branch_dataset/test/hblia0.smt2")
hard, obj, sense = inst.as_tuple()
stats = solve_omt_with_decider(hard, obj, sense)
print(stats["rlimit"])