import sys
import math
import tensorflow as tf

I = tf.linalg.LinearOperatorIdentity(2) #tf.constant(((1, 0), (0, 1)), dtype=tf.float32)
X = tf.constant(((0, 1), (1, 0)), dtype=tf.float32)
Z = tf.constant(((1, 0), (0, -1)), dtype=tf.float32)
H = tf.linalg.LinearOperatorFullMatrix(
    ((0.7071067811865475, 0.7071067811865475), (0.7071067811865475, -0.7071067811865475)))
CNOT = tf.linalg.LinearOperatorFullMatrix(
    ((1.0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)))
qbit = lambda alpha, name: tf.Variable([math.cos(alpha), math.sin(alpha)], name=name)

def main():
    # Define the qbit to teleport
    qC = qbit(0.6, 'C')

    # Make an EPR pair
    qA = qbit(1.5707963267948966, 'A')
    qB = qbit(0.0, 'B')
    
    hA = H.matvec(qA)
    hAB = tf.reshape((hA[..., None] * qB[None, ...]), (-1,))
    chAB = CNOT.matvec(hAB)
    cChA = CNOT.matvec(tf.reshape((qC[..., None] * hA[None, ...]), (-1,)))
    hC = H.matvec(qC)

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(chAB))
    # Alice measures

    # Bob receives the qubit


if __name__ == "__main__":
    main()