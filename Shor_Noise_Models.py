import os
import numpy as np

os.chdir("path/to/folder")

print('\nShor Code')
print('--------------')

from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit import QuantumCircuit #, transpile #,IBMQ
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
#from qiskit.tools.monitor import job_monitor

# Import from Qiskit Aer noise module
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, amplitude_damping_error,
                              pauli_error, depolarizing_error, thermal_relaxation_error)

#IBMQ.enable_account(‘ENTER API KEY HERE')
#provider = IBMQ.get_provider(hub='ibm-q')

#backend = provider.get_backend('ibmq_qasm_simulator')

sim_ideal = AerSimulator()

q = QuantumRegister(1,'q')
c = ClassicalRegister(1,'c')

simpl_circuit = QuantumCircuit(q,c)

simpl_circuit.h(q[0])

####error here############
simpl_circuit.x(q[0])#Bit flip error
simpl_circuit.z(q[0])#Phase flip error
############################

simpl_circuit.h(q[0])

simpl_circuit.barrier(q)

simpl_circuit.measure(q[0],c[0])

simpl_circuit.draw(output='mpl',filename='ideal_shorcircuit.png') #Draws an image of the circuit

job = sim_ideal.run(simpl_circuit, shots=1000)

#job_monitor(job)

counts = job.result().get_counts()

#Plot the histogram
fig = plot_histogram(counts)
fig.savefig('uncorrected_shor_output_histogram.png', format='png')

print("\n Uncorrected bit flip and phase error")
print("--------------------------------------")
print(counts)

#####Shor code starts here ########
q = QuantumRegister(9,'q')
c = ClassicalRegister(1,'c')

circuit = QuantumCircuit(q,c)

circuit.cx(q[0],q[3])
circuit.cx(q[0],q[6])

circuit.h(q[0])
circuit.h(q[3])
circuit.h(q[6])

circuit.cx(q[0],q[1])
circuit.cx(q[3],q[4])
circuit.cx(q[6],q[7])

circuit.cx(q[0],q[2])
circuit.cx(q[3],q[5])
circuit.cx(q[6],q[8])

circuit.barrier(q)

####error here############
circuit.x(q[0])#Bit flip error
circuit.z(q[0])#Phase flip error
############################

circuit.barrier(q)
circuit.cx(q[0],q[1])
circuit.cx(q[3],q[4])
circuit.cx(q[6],q[7])

circuit.cx(q[0],q[2])
circuit.cx(q[3],q[5])
circuit.cx(q[6],q[8])

circuit.ccx(q[1],q[2],q[0])
circuit.ccx(q[4],q[5],q[3])
circuit.ccx(q[8],q[7],q[6])

circuit.h(q[0])
circuit.h(q[3])
circuit.h(q[6])

circuit.cx(q[0],q[3])
circuit.cx(q[0],q[6])
circuit.ccx(q[6],q[3],q[0])

circuit.barrier(q)

circuit.measure(q[0],c[0])

circuit.draw(output='mpl',filename='corrected_shorcircuit.png') #Draws an image of the circuit

job = sim_ideal.run(circuit, shots=1000)

#job_monitor(job)

counts = job.result().get_counts()


#Plot the histogram
fig = plot_histogram(counts)
fig.savefig('corrected_shor_output_histogram.png', format='png')

print("\nShor code with bit flip and phase error")
print("----------------------------------------")
print(counts)


print("\nNoise Modelled Shor code with bit flip and phase error")
print("----------------------------------------")

# T1 and T2 values for qubits 0-3
T1s = np.random.normal(50e3, 10e3, 4) # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(70e3, 10e3, 4)  # Sampled from normal distribution mean 50 microsec

# Truncate random T2s <= T1s
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)])

# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_cx = 300
time_reset = 1000  # 1 microsecond
time_measure = 1000 # 1 microsecond

# QuantumError objects
errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                for t1, t2 in zip(T1s, T2s)]
errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                  for t1, t2 in zip(T1s, T2s)]
errors_u1  = [thermal_relaxation_error(t1, t2, time_u1)
              for t1, t2 in zip(T1s, T2s)]
errors_u2  = [thermal_relaxation_error(t1, t2, time_u2)
              for t1, t2 in zip(T1s, T2s)]
errors_u3  = [thermal_relaxation_error(t1, t2, time_u3)
              for t1, t2 in zip(T1s, T2s)]
errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
    thermal_relaxation_error(t1b, t2b, time_cx))
    for t1a, t2a in zip(T1s, T2s)]
    for t1b, t2b in zip(T1s, T2s)]

# Add errors to noise model
noise_thermal = NoiseModel()
for j in range(4):
    noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
    noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
    noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(4):
        noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])

print(noise_thermal)


# Run the noisy simulation
sim_thermal = AerSimulator(noise_model=noise_thermal)

# Transpile circuit for noisy basis gates
circ_tthermal = sim_thermal.run(simpl_circuit, shot=1000)

# Run and get counts
result_thermal = circ_tthermal.result()
counts_thermal = result_thermal.get_counts(0)

# Plot noisy output
plot_histogram(counts_thermal)



# Run the noisy simulation

# Transpile circuit for noisy basis gates
circ_tthermal = sim_thermal.run(simpl_circuit, shot=1000)


# Run and get counts
result_thermal = circ_tthermal.result()
counts_thermal = result_thermal.get_counts()

# Plot noisy output
plot_histogram(counts_thermal)

#Save the histogram
fig = plot_histogram(counts_thermal)
fig.savefig('notcorrected_shor_noisy_env_output_histogram.png', format='png')


# Run and get counts
result_thermal = sim_thermal.run(circuit).result()
counts_thermal = result_thermal.get_counts()

# Plot noisy output
plot_histogram(counts_thermal)

#Save the histogram
fig = plot_histogram(counts_thermal)
fig.savefig('corrected_shor_noisy_env_output_histogram.png', format='png')



T1 = 25.0
gate_time = 0.1
param_amp = 1-np.exp(-gate_time/T1)

error = amplitude_damping_error(param_amp)

amplitude_damping_noise_model = NoiseModel()
amplitude_damping_noise_model.add_all_qubit_quantum_error(error, 'h')

# Run the noisy simulation
sim_amplitude_damping = AerSimulator(noise_model= amplitude_damping_noise_model)


circ_tthermal = sim_amplitude_damping.run(simpl_circuit, shot=1000)


# Run and get counts
result_thermal = circ_tthermal.result()
counts_thermal = result_thermal.get_counts()

# Plot noisy output
plot_histogram(counts_thermal)

#Save the histogram
fig = plot_histogram(counts_thermal)
fig.savefig('notcorrected_shor_amplitude_damping_noisy_env_output_histogram.png', format='png')


# Run and get counts
result_thermal = sim_amplitude_damping.run(circuit).result()
counts_thermal = result_thermal.get_counts()

# Plot noisy output
plot_histogram(counts_thermal)

#Save the histogram
fig = plot_histogram(counts_thermal)
fig.savefig('corrected_shor_amplitude_damping_noisy_env_output_histogram.png', format='png')


#Construct the error for depolarizing error
depolarizing_error = depolarizing_error(0.1, 1) # 0.1 is the probability of error, 1 is the number of qubits

#Building noise model for depolarizing error
depolarizing_noise_model = NoiseModel()
depolarizing_noise_model.add_all_qubit_quantum_error(depolarizing_error, ['x', 'y', 'z'])

sim_depolarizing = AerSimulator(noise_model= depolarizing_noise_model)


# Run the noisy simulation
circ_depolarizing = sim_depolarizing.run(simpl_circuit, shot=1000)


# Run and get counts
result_depolarizing = circ_depolarizing.result()
counts_depolarizing = result_depolarizing.get_counts()

# Plot noisy output
plot_histogram(counts_depolarizing)

#Save the histogram
fig = plot_histogram(counts_depolarizing)
fig.savefig('notcorrected_shor_depolarizing_noisy_env_output_histogram.png', format='png')


# Run and get counts
result_depolarizing = sim_depolarizing.run(circuit, shot=1000).result()
counts_depolarizing = result_depolarizing.get_counts()

# Plot noisy output
plot_histogram(result_depolarizing)

#Save the histogram
fig = plot_histogram(result_depolarizing)
fig.savefig('corrected_shor_depolarizing_noisy_env_output_histogram.png', format='png')


#Combining thermal_noise, depolarization and amplitude damping noise models
combined_noise_instructions = noise_thermal.noise_instructions + depolarizing_noise_model.noise_instructions + amplitude_damping_noise_model.noise_instructions
combined_noise_model = NoiseModel(combined_noise_instructions)


sim_combined = AerSimulator(noise_model= combined_noise_model)


# Run the noisy simulation
circ_combined = sim_combined.run(simpl_circuit, shot=1000)


# Run and get counts
result_combined = circ_combined.result()
counts_combined = result_depolarizing.get_counts()

# Plot noisy output
plot_histogram(counts_combined)

#Save the histogram
fig = plot_histogram(counts_combined)
fig.savefig('notcorrected_shor_combined_noisy_env_output_histogram.png', format='png')


# Run and get counts
counts_combined = circ_combined.run(circuit, shot=1000).result()
counts_combined = result_depolarizing.get_counts()

# Plot noisy output
plot_histogram(counts_combined)

#Save the histogram
fig = plot_histogram(counts_combined)
fig.savefig('corrected_shor_combined_noisy_env_output_histogram.png', format='png')


