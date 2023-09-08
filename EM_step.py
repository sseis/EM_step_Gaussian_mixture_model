import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 2: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # (1) Initialize mu and sigma by splitting the n_examples data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)

    
    ######################################################################################################
    
    # split unlabelled data randomly into K Gaussian categories used in run_EM function (unpservised EM)
    # calculate mu and sigma for each random K group, mu shape = (K, n_features), cov shape = (K, n_feautres, n_featuers)
    # initialize phi at 1/K probability for each K Gaussian (K,) -- each phi is 0.25, will update through EM
    # w weights are initally 1's (n, K) - will update through EM
    
    # m = n_examples, i.e., number of unlabelled examples in entire data set
    
    ######################################################################################################
    
    # split data into 4 random Gaussian groups
    n, n_features = x.shape # (980, 2) 

    #------------------------------------ initialize params ----------------------------------------------
    # randomly seleteced means 
    mu = x[np.random.choice(n, K, False), :]
    # mu = mu.tolist()
    
    # covariance square matrices, (K, n_feautres, n_featuers) = (4,2,2)
    sigma = [np.eye(n_features)]*K

    # posterior probabilities 
    # [0.25 0.25 0.25 0.25], these phis will update through EM.  (K,) = (4,)
    phi = np.array([1/K]*K) 
    
    # responsibility probability matrix
    # for each data point for each of K Gaussians
    # (n,k) = (n,4), updated probabilities that each data point x_i belongs to each of K Gaussian cluster
    w = np.empty([n,K], dtype=float) #(980, 4)
    



    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(n)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (n_examples, dim).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.

    
         
        k = len(mu)  # n_clusters
        n, dim = x.shape
        # likelihoods = [] #log likelihoods
        
        # probability density function for multivariate normal distribution
        # posterior probability Q(z=j)
        pdf = lambda mu, sigma: np.linalg.det(sigma) ** -.5 ** (2 * np.pi) ** (-x.shape[1]/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                        x - mu, np.dot(np.linalg.inv(sigma) , (x - mu).T).T ) )
        
        # mu = np.array(mu) # turn mu list into an array
        
        prev_ll = ll
        
        it += 1
        
        ######################################################################################################
        
        # E-Step
        
        # n_unlabelled_examples
        
        # calculate likelihood of each observation x_i using the estimated parameters, mu and sigma
        # calculate likehlihood of x_i belonging to the jth K-cluster
        # use Bayes Theorem to get the posterior probability of kth Gaussian. (multivariate pdf)
        # phis were prior beliefs that an example was drawing from one of the Guassians - they were initiallized equally.
        # the phis add up to 1
        # we are refining these phis at each iteration until convergeance
        
        ######################################################################################################
        for i in range(k):

            w[:, i] = phi[i] * pdf(mu[i], sigma[i]) # posterior probability Q(z=j)
        
        
        ######################################################################################################
        
        # M-step
        
        # re-estimate our learning parameters, mus, sigmas, phis for each K Gaussian category.
        # To update the mean, weight each observation using conditional probabilities.
        # update until the updates are smaller than the given threshold, eps.
        
        # cluster with highest porbablity in the final E-step will indicate cluster assignments
        
        # update mu and sigma, phi and weights are constant
        
        ######################################################################################################
        # log likelihood 
        ll = np.sum(np.log(np.sum(w, axis = 1)))
        # print('current ll:', ll)
        # likelihoods.append(ll)

        w = (w.T/np.sum(w, axis=1)).T # normalize/ marginalize the responsibility matrix
        
        
        ######################################################################################################
        
        # M-step
        
        # re-estimate our learning parameters, mus, sigmas, phis for each K Gaussian category.
        # To update the mean, weight each observation using conditional probabilities.
        # update until the updates are smaller than the given threshold, eps.
        
        # cluster with highest porbablity in the final E-step will indicate cluster assignments
        
        # update mu and sigma, phi and weights are constant
        
        ######################################################################################################
        
        w_total = np.sum(w, axis=0) # datapoints in each K cluster

        for i in range(k):
            # update phi
            # phi[i] = (w_total[i]/(n))
            
            # new means for each k guassian
            mu[i] = 1/ w_total[i]*np.sum(w[:, i]*x.T, axis=1).T
                        
            # new covariance for each k guassian
            x_mus = np.matrix(x-mu[i]) # part of numerator
            sigma[i] = np.array(1/w_total[i]*np.dot(np.multiply(x_mus.T, w[:,i]), x_mus))
            
            w[i] = 1/n*w_total
        
        
        
        ######################################################################################################
        # Compute the log-likelihood of the data to check for convergence.
        # ll = sum_x[log(sum_z[p(x|z) * p(z)])]
        ######################################################################################################
        
        # print('likelihoods len:', len(likelihoods)) #list
        # print('likelihoods:', likelihoods)
        # if np.abs(ll - likelihoods[-2]) < eps: break
        # if np.abs(ll - prev_ll) < eps: break
        
        # print('last prev_ll:', prev_ll)
        
        print('iteration:', it)
   


        
   

    return w


def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, max_iter=1000):
    """Problem 2(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # (1) E-step: Update your estimates in w
        # (2) M-step: Update the model parameters phi, mu, and sigma
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        
        # *** START CODE HERE ***

        prev_ll = ll
        
        it += 1

        #------------------------------------------ E-Step ---------------------------------------------------
        #unlabeled
        w = e_step_unsupervised(phi, mu, sigma, x, w, K)
                
        #labeled
        w_tilde = w_labeled(x_tilde, z_tilde, K, alpha)
        
        #------------------------------------------ M-Step ---------------------------------------------------
        # log likelihoods

        #unlabeled
        ll_unlabeled= log_likelihood(w)

        #labeled
        ll_labeled = log_likelihood(w_tilde)
        
        ll = ll_unlabeled + ll_labeled*alpha
        
        #------------------------------------------ M-Step ---------------------------------------------------
        print('it:', it, 'phi sum:', np.sum(phi))
 
        
        # parameters update for each K Gaussian category
        phi, mu, sigma = m_step_semi_supervised(w, w_tilde, x , x_tilde, phi, mu, sigma, alpha)

        print('it:', it, 'phi update sum:', np.sum(phi))
        
        print('semi iteration:', it)


        # *** END CODE HERE ***
 

    return w



def e_step_unsupervised(phi, mu, sigma, x, w, K):
    # multivariate normal pdf for unlabeled
    pdf = lambda mu, sigma: np.linalg.det(sigma) ** -0.5 ** (2 * np.pi) ** (-x.shape[1]/2.0) \
            * np.exp(-0.5 * np.einsum('ij, ij -> i',\
                            x - mu, np.dot(np.linalg.inv(sigma) , (x - mu).T).T ) )
    # responsibility matrix (Q(z=k Gaussian)) = posterior probability
    for j in range(K):
        w[:, j] = phi[j] * pdf(mu[j], sigma[j]) 
    
    w= (w.T/np.sum(w, axis=1)).T 
    
    return w
    
def w_labeled(x_tilde, z_tilde, K, alpha):
    n_labeled = len(x_tilde) #20
    w_tilde = np.zeros([n_labeled,K], dtype=float) #(20, 4)
    
    for i in range(len(x_tilde)):
        for j in range(K):
            if int(z_tilde.flatten().tolist()[i]) == j:
                w_tilde[i, j] = alpha
            else:
                w_tilde[i, j] = 0
    
    return w_tilde

def log_likelihood(w):

    ll = np.sum(np.log(np.sum(w, axis = 0)))
    
    return ll


def m_step_semi_supervised(w, w_tilde, x , x_tilde, phi, mu, sigma, alpha):
    n_unlabeled = len(w) #980
    n_labeled = len(w_tilde) #20
    
    w_k_ex = np.sum(w, axis=0)
    w_k_ex_lab = np.sum(w_tilde, axis=0)*alpha
    
    # w_k_total = np.sum(w, axis=0) # unlabeled w data points per k
    w_k_total_lab = np.sum(w_tilde, axis=0) # labeled w data points per k
    
    w_tilde_alpha = w_tilde*alpha
 
    x_total = np.concatenate((x, x_tilde), axis = 0) # unlabeled + labeled
    
    w_total = np.concatenate((w, w_tilde_alpha), axis = 0) # unlabeled + labeled (1000, 4)
    
    w_k_total = np.sum(w_total, axis=0) # unlabeled + labeled w data points per k  (4,)
    
    # print('w_total:', w_total)
    # print('w_total shape:', w_total.shape)
    print('w_k_total:', w_k_total)
    print('w_k_total shape:', w_k_total.shape)
    
    

            
    for j in range(K):
        phi[j] = (w_k_total[j])/(n_labeled + n_unlabeled)
        
        # new means for each k guassian
        mu[j] = 1/ w_k_total[j]*np.sum(w_total[:, j]*x_total.T, axis=1).T
                        
        # new covariance for each k guassian
        x_total_mus = np.matrix(x_total-mu[j]) # part of numerator
        
        sigma[j] = np.array(1/w_k_total[j]*np.dot(np.multiply(x_total_mus.T, w_total[:,j]), x_total_mus))


    return phi, mu, sigma




def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)
        main(is_semi_supervised=True, trial_num=t)