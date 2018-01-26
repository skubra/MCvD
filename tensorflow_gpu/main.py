#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf
import math
import numpy
import time
from tensorflow.python.client import timeline


# Configuration Variables

N = 10000

d = 4
r_rcv = 4       # receiver radius
r_mm = 0.001    # molecule radius
r_xmt = 4       # transmitter radius

n_rcv = 3       # n of receivers
n_xmt = 2       # n of transmitters

D = 79.4

sim_t = 5.0     # run time
t = 0.001       # step time
t_i = 0.001     # iterator

sigma = math.sqrt(2 * D * t)

# Tensors

sigma2              = tf.constant(sigma*sigma)
receivers_tensor    = tf.get_variable("receivers_tensor",       (n_rcv, n_xmt, N, 3),   initializer=tf.zeros_initializer)
transmitters_tensor = tf.get_variable("transmitters_tensor",    (n_xmt, n_xmt, N, 3),   initializer=tf.zeros_initializer)
particles_tensor    = tf.get_variable("particles_tensor",       (n_xmt, N, 3),          initializer=tf.zeros_initializer)
random_tensor       = tf.get_variable("random_tensor",          (n_xmt, N, 3))
flag_tensor         = tf.get_variable("flag_tensor",            (n_xmt, N, 1),          initializer=tf.zeros_initializer(), dtype=tf.bool)
inverse_tensor      = tf.get_variable("inverse_tensor",         (n_xmt, N, 1),          initializer=tf.ones_initializer(), dtype=tf.bool)
received_by_i       = tf.get_variable("received_by_i",          n_rcv,                  initializer=tf.zeros_initializer(), dtype=tf.int32)
distance_to_rcv     = tf.get_variable("distance_to_rcv",        (n_xmt, N, 1),          initializer=tf.zeros_initializer(), dtype=tf.bool)


def initialize_tensors():
    if n_xmt % 2 == 1:
        sess.run(tf.scatter_add(transmitters_tensor, 0, numpy.tile(numpy.tile(numpy.array([[0.0, 0.0, 0.0]]), (N, 1)), (n_xmt, 1, 1))))
        sess.run(tf.scatter_add(particles_tensor, 0, numpy.tile(numpy.array([[0.0 + r_xmt, 0.0, 0.0]]), (N, 1))))
        for i in range(1, n_xmt / 2 + 1):
            sess.run(tf.scatter_add(transmitters_tensor, i*2-1, numpy.tile(numpy.tile(numpy.array([[0.0, (2 * r_xmt + d / 2.0) * i, 0.0]]), (N, 1)), (n_xmt, 1, 1))))
            sess.run(tf.scatter_add(transmitters_tensor, i*2, numpy.tile(numpy.tile(numpy.array([[0.0, (2 * r_xmt + d / 2.0) * i * -1, 0.0]]), (N, 1)), (n_xmt, 1, 1))))
            sess.run(tf.scatter_add(particles_tensor, i*2-1, numpy.tile(numpy.array([[0.0 + r_xmt, (2 * r_xmt + d / 2.0) * i, 0.0]]), (N, 1))))
            sess.run(tf.scatter_add(particles_tensor, i*2, numpy.tile(numpy.array([[0.0 + r_xmt, (2 * r_xmt + d / 2.0) * i * -1, 0.0]]), (N, 1))))

    else:
        for i in range(0, n_xmt / 2):
            sess.run(tf.scatter_add(transmitters_tensor, i*2, numpy.tile(numpy.tile(numpy.array([[0.0, d / 4.0 + r_xmt + (2 * r_xmt + d / 2.0) * i, 0.0]]), (N, 1)), (n_xmt, 1, 1))))
            sess.run(tf.scatter_add(transmitters_tensor, i*2+1, numpy.tile(numpy.tile(numpy.array([[0.0, -1 * (d / 4.0 + r_xmt + (2 * r_xmt + d / 2.0) * i), 0.0]]), (N, 1)), (n_xmt, 1, 1))))
            sess.run(tf.scatter_add(particles_tensor, i*2, numpy.tile(numpy.array([[0.0 + r_xmt, d / 4.0 + r_xmt + (2 * r_xmt + d / 2.0) * i, 0.0]]), (N, 1))))
            sess.run(tf.scatter_add(particles_tensor, i*2+1, numpy.tile(numpy.array([[0.0 + r_xmt, -1 * (d / 4.0 + r_xmt + (2 * r_xmt + d / 2.0) * i), 0.0]]), (N, 1))))

    if n_rcv % 2 == 1:
        temp = numpy.array([[r_rcv + r_xmt + d, 0.0, 0.0]])
        temp = numpy.tile(numpy.tile(temp, (N, 1)), (n_xmt, 1, 1))
        sess.run(tf.scatter_add(receivers_tensor, 0, temp))
        for i in range(1, n_rcv / 2 + 1):
            temp = numpy.array([[r_rcv + r_xmt + d, (2 * r_rcv + d / 2.0) * i, 0.0]])
            temp = numpy.tile(numpy.tile(temp, (N, 1)), (n_xmt, 1, 1))
            sess.run(tf.scatter_add(receivers_tensor, i*2-1, temp))
            temp = numpy.array([[r_rcv + r_xmt + d, -1 * (2 * r_rcv + d / 2.0) * i, 0.0]])
            temp = numpy.tile(numpy.tile(temp, (N, 1)), (n_xmt, 1, 1))
            sess.run(tf.scatter_add(receivers_tensor, i*2, temp))

    else:
        for i in range(0, n_rcv / 2):
            temp = numpy.array([[r_rcv + r_xmt + d, d / 4.0 + r_rcv + (2 * r_rcv + d / 2.0) * i, 0.0]])
            temp = numpy.tile(numpy.tile(temp, (N, 1)), (n_xmt, 1, 1))
            sess.run(tf.scatter_add(receivers_tensor, i*2, temp))
            temp = numpy.array([[r_rcv + r_xmt + d, -1 * (d / 4.0 + r_rcv + (2 * r_rcv + d / 2.0) * i), 0.0]])
            temp = numpy.tile(numpy.tile(temp, (N, 1)), (n_xmt, 1, 1))
            sess.run(tf.scatter_add(receivers_tensor, i*2+1, temp))


# Placeholders

X_i = tf.placeholder(numpy.int32)
X_q = tf.placeholder(numpy.int32)

# Tensor Operations

init = tf.global_variables_initializer()
# Generate random numbers and assign it to the random tensor
random_generator = tf.assign(random_tensor, tf.random_normal((n_xmt, N, 3), 0, sigma2))

# Add random numbers to the particle tensor
assign_add_random_to_particles = tf.assign_add(particles_tensor, random_generator)

# Calculate distance between receiver i and particles, then check the particles received
dist1_ = tf.assign(distance_to_rcv, tf.less_equal(tf.sqrt(tf.reduce_sum(tf.squared_difference(particles_tensor, receivers_tensor[X_i]), 2, keep_dims=True)), r_rcv + r_mm))

# Calculate number of received particles by ith receiver
received_by_ith_receiver = tf.count_nonzero(tf.logical_and(tf.logical_xor(flag_tensor, inverse_tensor), dist1_))

# Update each receiver by number of received particles
update_receiver = tf.scatter_add(received_by_i, X_i, tf.cast(received_by_ith_receiver, tf.int32))

# Update flag according to total received particles
update_flag = tf.assign(flag_tensor, tf.logical_or(distance_to_rcv, flag_tensor))

# Calculate distance between transmitter i and particles, then check the particles bounced
dist2_ = tf.cast(tf.less_equal(tf.sqrt(tf.reduce_sum(tf.squared_difference(particles_tensor, transmitters_tensor[X_i]), 2, keep_dims=True)), r_xmt + r_mm), tf.float32)

# Put the particles that touched transmitter to their old places
bounce_from_xmt = tf.assign_sub(particles_tensor, tf.multiply(random_tensor, tf.tile(dist2_, (1, 1, 3))))

with tf.Session() as sess:

    sess.run(init)
    initialize_tensors()

    writer = tf.summary.FileWriter('./graphs', sess.graph)
    start_op = time.time()

    while t_i < sim_t:

        sess.run(assign_add_random_to_particles)

        # Check if the particle is inside of each receiver

        for i in range(0, n_rcv):

            # To print the receiving times
            # for k in range(sess.run(received_by_ith_receiver, feed_dict={X_i: i})):
            #     print t_i

            sess.run(update_receiver, feed_dict={X_i: i})
            sess.run(update_flag)

        # Reflection on the surface of each transmitter

        for i in range(0, n_xmt):
            sess.run(bounce_from_xmt, feed_dict={X_i: i})

        t_i += t

    end_op = time.time()

    print 'time: ', end_op - start_op
    print sess.run(tf.count_nonzero(flag_tensor))
    print sess.run(received_by_i)

writer.close()

