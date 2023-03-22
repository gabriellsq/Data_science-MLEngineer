import numpy as np

# define transition matrix
transition_matrix = np.array([
    [0.2, 0.4, 0.4],  # transition probabilities from A
    [0.3, 0.1, 0.6],  # transition probabilities from B
    [0.1, 0.3, 0.6],  # transition probabilities from C
    #[0.1, 0.2, 0.5],  # transition probabilities from D
    # add more rows for other events (D through Z)
])

# define initial state probabilities
initial_state = np.array([0.4, 0.3, 0.3])  # initial probabilities for A, B, and C

# define the state sequence
states = ["A", "B", "C"]  # add more events (D through Z) as needed

# define the event we want to calculate the probability for
event = "A"

# get the index of the event in the state sequence
event_index = states.index(event)

# calculate the probability of the event using the initial state and transition matrix
probability = np.dot(np.linalg.matrix_power(transition_matrix, 1), initial_state)[event_index]

print(f"The probability of event {event} happening is: {probability:.2f}")
