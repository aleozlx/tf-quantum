import sys
import math
import tensorflow as tf

I = tf.linalg.LinearOperatorIdentity(2)
X = tf.linalg.LinearOperatorFullMatrix(((0.0, 1), (1, 0)))
Z = tf.linalg.LinearOperatorFullMatrix(((1.0, 0), (0, -1)))
H = tf.linalg.LinearOperatorFullMatrix(
    ((0.7071067811865475, 0.7071067811865475), (0.7071067811865475, -0.7071067811865475)))
# TODO forgot to factorize after cnot??
CNOT = tf.linalg.LinearOperatorFullMatrix(
    ((1.0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0)))
qbit = lambda alpha, name: tf.Variable([math.cos(alpha), math.sin(alpha)], name=name)

def outer(a, b):
    return tf.reshape((a[..., None] * b[None, ...]), (-1,))

def makeEPR(*names):
    qA = qbit(1.5707963267948966, names[0])
    hA = H.matvec(qA)
    qB = qbit(0.0, names[1])
    return hA, CNOT.matvec(outer(hA, qB))

def measure(q, sess):
    return 0

def encode(q, hA, sess):
    hC = H.matvec(q)
    cChA = CNOT.matvec(outer(q, hA))
    return (measure(cChA, sess) << 1) | measure(hC, sess)

def main():
    # Make an EPR pair; Alice will have hA and Bob will have chAB
    hA, chAB = makeEPR('qA', 'qB')

    # Alice encodes an arbitrary qbit
    qC = qbit(0.6, 'qC') # the qbit to teleport

    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        m = encode(qC, hA, sess)

    # Bob decodes
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(chAB))

if __name__ == "__main__":
    main()