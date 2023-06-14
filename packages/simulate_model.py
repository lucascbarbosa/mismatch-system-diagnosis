import numpy as np
import pandas as pd


class Simulator(object):
    def __init__(
        self,
        n_cvs,
        n_mvs,
        h,
        sys_span,
        noise_span,
        simualtion_dir,
    ):
        self.n_cvs = n_cvs
        self.n_mvs = n_mvs
        self.h = h
        self.sys_span = sys_span
        self.noise_span = noise_span
        self.simualtion_dir = simualtion_dir

    def is_stable(self, A):
        return np.all(np.real(np.linalg.eigvals(A) < 0))

    def create_system(self):
        A = np.random.uniform(-self.sys_span, 0, (self.n_cvs, self.n_mvs))
        A = np.diag(np.diag(A))
        offdiag = np.zeros((self.n_cvs, self.n_cvs))
        offdiag = np.random.uniform(-0.1, 0.1, (self.n_cvs, self.n_cvs))
        offdiag -= np.diag(np.diag(offdiag))
        A = np.diag(A) + offdiag
        while not self.is_stable(A):
            damping_coeff = 0.5  # choose a damping coefficient
            A -= damping_coeff * np.eye(self.n_cvs)
        A = A * self.h + np.eye(self.n_cvs)
        B = np.random.random((self.n_cvs, self.n_mvs)) * self.h
        return A.round(3), B.round(3)

    def create_ctrl(self, N):
        u = np.random.normal(size=(N, self.n_mvs))
        return u

    def simulate(self, N, y0, mismatch_ratio, history, noise=True):
        u_data = self.create_ctrl(N)
        y = np.zeros((N, self.n_cvs))
        y[0, :] = y0

        A, B = self.create_system()
        A_hat, B_hat = A.copy(), B.copy()
        mismatch = False
        mismatch_count = 0
        match_count = 0
        mismatch_log = np.zeros((N, 1))

        real_models = [(A, B)]
        estimated_models = [(A_hat, B_hat)]

        for i in range(1, N):
            y_row = y[i - 1].reshape((self.n_cvs, 1))
            u = u_data[i - 1].reshape((self.n_mvs, 1))
            y[i, :] = np.dot(A, y_row).T + np.dot(B, u).T
            if noise:
                y[i, :] += np.random.normal(loc=0.0, scale=self.noise_span)

            # saturate
            y[i, :] = np.clip(y[i, :], a_min=-1e6, a_max=1e6)

            rand = np.random.random()
            if (
                rand < mismatch_ratio
                and not mismatch
                and match_count >= 5 * history
            ):
                mismatch = True
                match_count = 0
                A, B = self.create_system()
                real_models.append((A, B))
            if (
                rand > mismatch_ratio
                and mismatch
                and mismatch_count >= 5 * history
            ):
                mismatch = False
                mismatch_count = 0
                A_hat, B_hat = A.copy(), B.copy()
                estimated_models.append((A_hat, B_hat))
                y[i, :] = y[i - 1, :]

            if mismatch:
                mismatch_count += 1
            else:
                match_count += 1

            mismatch_log[i] = int(mismatch)

        return u_data, y, mismatch_log, estimated_models

    def get_fir_coefs(self, model, N):
        A, B = model
        coefs = np.zeros((self.n_cvs, self.n_mvs, N + 1))
        for i in range(self.n_mvs):
            for j in range(self.n_cvs):
                y = np.zeros((N + 1, self.n_cvs))
                u = np.zeros((self.n_mvs, 1))
                u[i, 0] = 1
                for k in range(1, N + 1):
                    y_row = y[k - 1, :].reshape((self.n_cvs, 1))
                    y[k, :] = np.dot(A, y_row).T + np.dot(B, u).T
                    coefs[:, j, k] = y[k, :]

        coefs_lag = coefs[:, :, :-1]
        coefs = coefs[:, :, 1:]
        return coefs - coefs_lag

    def create_data(self, output_idx, output, inputs):
        data = pd.DataFrame()
        time = pd.date_range(
            start=0, periods=output.shape[0], freq="20T"
        ).to_numpy()
        data["time"] = time

        output = output[:, output_idx]
        data[f"CV_{output_idx+1}"] = output

        data[[f"MV_{i+1}" for i in range(self.n_mvs)]] = inputs
        return data
