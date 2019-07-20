import sys
import math
import tensorflow as tf

I = tf.constant(((1, 0), (0, 1)), dtype=tf.float32)
X = tf.constant(((0, 1), (1, 0)), dtype=tf.float32)
Z = tf.constant(((1, 0), (0, -1)), dtype=tf.float32)
H = tf.constant(((0.7071067811865475, 0.7071067811865475), (0.7071067811865475, -0.7071067811865475)), dtype=tf.float32)
CNOT = tf.constant(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)), dtype=tf.float32)
qbit = lambda alpha, name: tf.Variable((math.cos(alpha), math.sin(alpha)), name=name)

def main():
    # Define the qbit to teleport
    qC = qbit(0.6, 'C')

    # Make an EPR pair
    qA = qbit(1.5707963267948966, 'A')
    qB = qbit(0.0, 'B')
    
    # H * qA
    sys.exit()

    # Alice measures

    # Bob receives the qubit


if __name__ == "__main__":
    main()