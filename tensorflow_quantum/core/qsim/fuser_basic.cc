/* Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_quantum/core/qsim/fuser_basic.h"

#include <iostream>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/src/circuit.h"

namespace tfq {
namespace qsim {
namespace {

using ::tensorflow::Status;

// Appends to `fused_gates` all single-qubit gates on the current `qubit_wire`,
// until reaching the end of the wire or a multi-qubit gate.
// Starts the search at `current_timeslice`.
void Advance(const std::vector<const Gate*>& qubit_wire, GateFused* gate,
             unsigned int* current_timeslice) {
  while (*current_timeslice < qubit_wire.size() &&
         qubit_wire[*current_timeslice]->num_qubits == 1) {
    gate->AddGate(qubit_wire[(*current_timeslice)++]);
  }
}

}  // namespace

GateFused::GateFused(const unsigned int time, const unsigned int q0,
                     const unsigned int q1, const Gate* anchor)
    : time_(time), anchor_(anchor) {
  qubits_[0] = q0;
  qubits_[1] = q1;
}

void GateFused::GetAllGates(const std::vector<const Gate*>*& ref) const {
  ref = &gates_;
}

unsigned int GateFused::GetNumGates() const { return gates_.size(); }

// Return a copy of the gate pointed to by the given index
const Gate* GateFused::GetGate(unsigned int gate_index) const {
  return gates_.at(gate_index);
}

void GateFused::SetGate(unsigned int gate_index, const Gate* gate) {
  gates_.at(gate_index) = gate;
}

unsigned int GateFused::GetTime() const { return time_; }
void GateFused::SetTime(unsigned int time) { time_ = time; }

unsigned int GateFused::GetQubit0() const { return qubits_[0]; }
void GateFused::SetQubit0(unsigned int q0) { qubits_[0] = q0; }

unsigned int GateFused::GetQubit1() const { return qubits_[1]; }
void GateFused::SetQubit1(unsigned int q1) { qubits_[1] = q1; }

const Gate* GateFused::GetAnchor() const { return anchor_; }
void GateFused::SetAnchor(const Gate* anchor) { anchor_ = anchor; }

bool operator==(const GateFused& l, const GateFused& r) {
  if (l.GetTime() != r.GetTime()) {
    return false;
  }
  if (l.GetQubit0() != r.GetQubit0()) {
    return false;
  }
  if (l.GetQubit1() != r.GetQubit1()) {
    return false;
  }
  if (*l.GetAnchor() != *r.GetAnchor()) {
    return false;
  }
  if (l.GetNumGates() != r.GetNumGates()) {
    return false;
  }
  for (size_t i = 0; i < l.GetNumGates(); i++) {
    if (*l.GetGate(i) != *r.GetGate(i)) {
      return false;
    }
  }
  return true;
}

bool operator!=(const GateFused& l, const GateFused& r) { return !(l == r); }

Status FuseGates(const Circuit& circuit, std::vector<GateFused>* fused) {
  // Holds only the two-qubit gates in the circuit, in correct time order;
  // these are later used as anchors for single qubit gate fusing.
  std::vector<const Gate*> gates_seq;
  gates_seq.resize(0);
  gates_seq.reserve(circuit.gates.size());

  // Lattice of gates. The first index refers to a qubit location;
  // the second index refers to a time slice of simultaneously occurring gates.
  std::vector<std::vector<const Gate*>> gates_lat(circuit.num_qubits);

  // Reserve 128 time slices for each qubit in the lattice.
  for (unsigned int k = 0; k < circuit.num_qubits; ++k) {
    gates_lat[k].resize(0);
    gates_lat[k].reserve(128);
  }

  // Examine every gate in the circuit.
  // Place a reference in gates_lat at all lattice points (locations and times)
  // at which the gate acts; record each two-qubit gate in gates_seq.
  for (const auto& gate : circuit.gates) {
    if (gate.num_qubits == 1) {
      gates_lat[gate.qubits[0]].push_back(&gate);
    } else if (gate.num_qubits == 2) {
      gates_lat[gate.qubits[0]].push_back(&gate);
      gates_lat[gate.qubits[1]].push_back(&gate);
      gates_seq.push_back(&gate);
    }
  }

  // For each qubit, holds the latest timeslice processed on that qubit.
  std::vector<unsigned int> last(circuit.num_qubits, 0);

  // Fuse gates.
  // Fusing is performed by having each two-qubit gate (anchor) in the sequence
  // greedily absorb all single-qubit gates around them.
  for (const Gate* pgate : gates_seq) {
    unsigned int q0 = pgate->qubits[0];
    unsigned int q1 = pgate->qubits[1];

    // No more unprocessed gates available on q0.
    if (last[q0] >= gates_lat[q0].size()) continue;

    // This two-qubit gate has already been absorbed into a different anchor.
    if (gates_lat[q0][last[q0]]->time > pgate->time) continue;

    GateFused gate_f(pgate->time, q0, q1, pgate);
    do {
      // Collect all available single-qubit gates before the anchor.
      Advance(gates_lat[q0], &gate_f, &last[q0]);
      Advance(gates_lat[q1], &gate_f, &last[q1]);

      // Initial fuse should end at the anchor which initiated the fuse.
      if (gates_lat[q0][last[q0]] != gates_lat[q1][last[q1]]) {
        return Status(tensorflow::error::INVALID_ARGUMENT,
                      "Error fusing gates.");
      }

      // Collect the anchor.
      gate_f.AddGate(gates_lat[q0][last[q0]]);

      // Collect all available single-qubit gates after the anchor.
      last[q0]++;
      last[q1]++;
      Advance(gates_lat[q0], &gate_f, &last[q0]);
      Advance(gates_lat[q1], &gate_f, &last[q1]);

    } while (
        // There are still gates available on both wires
        last[q0] < gates_lat[q0].size() &&
        last[q1] < gates_lat[q1].size()
        // The next gate is a two-qubit gate sharing both qubits with the anchor
        && gates_lat[q0][last[q0]] == gates_lat[q1][last[q1]]);

    fused->push_back(std::move(gate_f));
  }

  // TODO: deal with single-qubit orphan gates if present.
  // TODO: Add a check for single-qubits gates, and return error if present.

  return Status::OK();
}

}  // namespace qsim
}  // namespace tfq
