// 4-qubit SAFe service kernel, Ry+Rz + CNOT ladder, measure all.
// Defined in src/circuits/safe_kernel.py (see PAPER_AMENDMENTS A3).
OPENQASM 3.0;
include "stdgates.inc";
bit[4] c;
qubit[4] q;
ry(0.2) q[0];
rz(0.2) q[0];
ry(0.4) q[1];
rz(0.4) q[1];
ry(0.6000000000000001) q[2];
rz(0.6000000000000001) q[2];
ry(0.8) q[3];
rz(0.8) q[3];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
barrier q[0], q[1], q[2], q[3];
c[0] = measure q[0];
c[1] = measure q[1];
c[2] = measure q[2];
c[3] = measure q[3];
