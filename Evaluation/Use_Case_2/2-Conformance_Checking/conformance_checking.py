from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments_algo
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.objects.log.importer.xes import importer as xes_importer

# 1. Load the Petri net model (from .pnml file)
net, initial_marking, final_marking = pnml_importer.apply("../1-Reference_Model/reference_model.pnml")

# 2. Load the event log (from .xes file)
log = xes_importer.apply("nonconforming_event_log.xes")
# log = xes_importer.apply("../1-Reference_Model/Event log for reference model.xes")

# 3. Perform token-based replay
token_replay_results = token_replay.apply(log, net, initial_marking, final_marking)

# 4. Perform alignment-based conformance checking
alignment_results = alignments_algo.apply_log(log, net, initial_marking, final_marking)

# 5. Print conformance results for each trace
for idx, (token_result, align_result) in enumerate(zip(token_replay_results, alignment_results)):
    model_moves = []
    log_moves = []

    for move in align_result['alignment']:
        log_label, model_label = move[0], move[1]
        if log_label == '>>':
            model_moves.append(model_label)
        elif model_label == '>>':
            log_moves.append(log_label)

    print(f"Trace {idx + 1}:")
    # print("  [Token-based Replay]")
    # print("    Fitness:", token_result['trace_fitness'])
    # print("    Missing Tokens:", token_result['missing_tokens'])
    # print("    Remaining Tokens:", token_result['remaining_tokens'])
    # print("    Produced Tokens:", token_result['produced_tokens'])
    # print("    Consumed Tokens:", token_result['consumed_tokens'])

    print("  [Alignment-based Conformance]")
    print("    Fitness:", align_result['fitness'])
    print("    Model Moves (model only):", model_moves)
    print("    Log Moves (log only):", log_moves)
    print("    Alignment Steps:", align_result['alignment'])
