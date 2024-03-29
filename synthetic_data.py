import numpy as np
from scipy.stats import ortho_group
import config as c


class TaskData:
    '''Initializes data set and optimal bandit parameter, implicitly determining rewards as well.'''

    def __init__(self,
                 task_no=c.NO_TASK,
                 rank=c.DIMENSION_ALIGN,
                 dimension=c.DIMENSION,
                 context_size=c.CONTEXT_SIZE):

        self.dimension = dimension
        sigma_t = np.random.uniform(size=dimension)
        self.target_context = np.random.multivariate_normal(mean=np.zeros(dimension),
                                                            cov=np.diag(sigma_t),
                                                            size=context_size)
        self.rank = rank
        self.subspace_projection, ortho_mat = self.projection_matrix()
        # print(self.subspace_projection)
        # self.red_projection = self.reduced_projection(ortho_mat)
        if c.DIMENSION_ALIGN < c.DIMENSION:
            self.theta_opt = np.random.multivariate_normal(mean=np.zeros(dimension),
                                                           cov=c.VAR_MAX/(c.DIMENSION-c.DIMENSION_ALIGN)*np.identity(dimension),
                                                           size=task_no)
        else:
            self.theta_opt = np.random.multivariate_normal(mean=np.zeros(dimension),
                                                           cov=np.identity(dimension),
                                                           size=task_no)

        # norms = np.sqrt(np.einsum('ij,ij->i', self.theta_opt, self.theta_opt))
        # for i in np.arange(task_no):
        #     if norms[i] > c.V_MAX:
        #         self.theta_opt[i] *= c.V_MAX/norms[i]

        self.theta_opt = self.subspace_projection.dot(self.theta_opt.T).T
        self.off_set = np.dot(np.identity(dimension) - self.subspace_projection,
                              np.random.uniform(size=dimension)).T
        # if np.linalg.norm(self.off_set) > 1:
        #     self.off_set = self.off_set/np.linalg.norm(self.off_set)
        self.theta_opt += self.off_set
        if c.DIMENSION_ALIGN > 0:
            self.theta_opt += np.dot(np.identity(dimension) - self.subspace_projection,
                                     np.random.multivariate_normal(np.zeros(c.DIMENSION),
                                                                   c.KAPPA/c.DIMENSION_ALIGN*np.identity(c.DIMENSION),
                                                                   size=c.NO_TASK).T).T

        # if np.dot(self.theta_opt, self.theta_opt) > 1:
        #     self.theta_opt /= np.sqrt(np.dot(self.theta_opt, self.theta_opt))


    def projection_matrix(self):
        '''Generate random orthonormal projection given rank of the subspace.'''
        ortho_mat = ortho_group.rvs(dim=self.dimension)

        if self.rank != 0:
            ortho_mat[:, -self.rank:] = 0

        sub_proj = ortho_mat.dot(ortho_mat.T)
        return sub_proj, ortho_mat


    # def reduced_projection(self, ortho_mat):
    #     '''Generate lower rank projection matrix based on a given projection.'''
    #     if self.rank != 0:
    #         ortho_mat[:, -(self.rank + 1):] = c.KAPPA * np.ones((self.dimension, self.rank + 1))

    #     red_proj = ortho_mat.dot(ortho_mat.T)
    #     return red_proj

# TEST_CONTEXT = TaskData()
# TEST_CONTEXT.projection_matrix(2)
