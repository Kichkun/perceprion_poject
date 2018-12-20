import numpy as np
from numpy import dot
import numpy.linalg as lin
from numpy.linalg import cholesky
import scipy.sparse.linalg as sp_lin
from scipy.sparse import csr_matrix

from sam.tools.task import sample_from_odometry, get_motion_noise_covariance, wrap_angle, get_prediction, get_expected_observation
from sam.tools.task import Gt, Ht, Jt, h_inv

class SAM:
    def __init__(self, initial_pos, initial_sigma, alphas, Q):
        self.states = np.expand_dims(initial_pos, axis=0)
        self.actions = np.zeros((0, 3))
        self.observations = np.zeros((0, 4))
        self.landmarks = np.zeros((0, 2))

        self.alphas = alphas
        self.W0 = cholesky(lin.inv(initial_sigma))
        self.W_k = cholesky(lin.inv(Q[:2,:2]))

        self.t = 0
        self.landmarks_obs = 0
        self.obs_order = {}

    def predict(self, u):
        new_state = sample_from_odometry(self.states[-1], u, self.alphas)
        self.states = np.vstack((self.states, new_state))
        self.actions = np.vstack((self.actions, u))
        self.t += 1

    def update(self, z):
        for i in z:
            # Add new landmarks if any
            self.check_correspondence(i)

        # Add information from which position these observations were made
        obs_from = np.array([self.t] * z.shape[0]).reshape((z.shape[0], 1))
        self.observations = np.vstack((self.observations, np.hstack((obs_from, z))))

        # Problem is here
        A, b = self.create_A_and_b()

        # delta = sp_lin.spsolve(dot(A.T, A), dot(A.T, b))

        L = csr_matrix(cholesky(dot(A.T, A)))
        y = sp_lin.spsolve_triangular(L, dot(A.T, b), lower=True)
        delta = sp_lin.spsolve_triangular(L.T, y, lower=False)

        self.states += delta[:3 * (self.t + 1)].reshape(self.states.shape)
        if delta.shape[0] > 3 * (self.t + 1):
            self.landmarks += delta[3 * (self.t + 1):].reshape(self.landmarks.shape)

    def create_A_and_b(self):
        G_block = dot(self.W0, -1 * np.identity(3))
        a_block = np.zeros((3, 1))
        for i in np.arange(0, self.t):
            G = Gt(self.states[i], self.actions[i])
            R = get_motion_noise_covariance(self.actions[i], self.alphas)
            W_i = cholesky(lin.inv(R))

            G_block = np.block([[G_block, np.zeros((3 * (i + 1), 3))],
                                [np.zeros((3, 3 * i)), dot(W_i, G), dot(W_i, -1 * np.identity(3))]])

            a = self.states[i + 1] - get_prediction(self.states[i], self.actions[i])
            a[2] = wrap_angle(a[2])
            a_block = np.vstack((a_block, dot(W_i, a.reshape((3, 1)))))

        # Construct part of matrix (H_J_block) and residuals (c_block) for observation equations
        H_J_block = np.zeros((0, 3 * (self.t + 1) + 2 * self.landmarks_obs))
        c_block = np.zeros((0, 1))
        for i in self.observations:
            # Index of state at which observation was made
            s_pos = int(i[0])
            # Index of landmark in matrix
            l_pos = self.obs_order[i[3]] - 1

            state = self.states[s_pos]
            m_j = self.landmarks[l_pos]

            H = Ht(state, m_j)
            J = Jt(state, m_j)

            H_part = np.hstack((np.zeros((2, s_pos * 3)), dot(self.W_k, H), np.zeros((2, 3 * (self.t - s_pos)))))
            J_part = np.hstack(
                (np.zeros((2, l_pos * 2)), dot(self.W_k, J), np.zeros((2, 2 * (self.landmarks_obs - 1 - l_pos)))))
            H_J_block = np.vstack((H_J_block, np.hstack((H_part, J_part))))

            c = i[1:3] - get_expected_observation(state, m_j)
            c[1] = wrap_angle(c[1])
            c_block = np.vstack((c_block, dot(self.W_k, c.reshape((2, 1)))))

        # IF there is no observations then return only odometry part. Otherwise construct the whole matrix.
        if H_J_block.shape[0] == 0:
            return G_block, a_block
        else:
            return np.block([[G_block, np.zeros((3 * (self.t + 1), 2 * self.landmarks_obs))],
                             [H_J_block]]), \
                   np.vstack((a_block, c_block))

    def check_correspondence(self, z):
        correspondence = z[2]
        if correspondence not in self.obs_order:
            self.landmarks_obs += 1
            self.obs_order[correspondence] = self.landmarks_obs

            state = self.states[self.t]
            m_j = h_inv(state, z)
            self.landmarks = np.vstack((self.landmarks, m_j))

    def get_robot_position(self):
        return self.states

    def get_landmarks(self):
        return self.landmarks