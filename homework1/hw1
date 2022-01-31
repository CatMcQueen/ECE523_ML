import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import stats


def make_random_var(mu, sigma, n, d):
    rand = np.random.multivariate_normal(mu, sigma, size=d)
    return rand

def get_probability_score(x, mu, sigma, d):
    #print("X is {}, {}".format(x, sigma))
    exponent_thing = math.e**(-.5*1/(sigma**2) * (x-mu)**2)
    const_thing    = 1/((2*math.pi)**d/2 *sigma**(1/2))
    # print(exponent_thing, const_thing)
    return const_thing * exponent_thing

    #############################
    ## HW 1: 2.2 Discriminant
    #############################
def get_descriminant(x, mu, sigma, d, pwi):
    a = -.5*(x-mu).T
    b = np.linalg.inv(sigma)
    c = (x-mu)
    firstterm   = np.dot(a, np.dot(b,c))#np.dot(np.dot(a, b), c)
    secondterm  = (d*.5)*np.log(2*math.pi)
    thirdterm   = .5 * np.log(np.linalg.det(sigma))
    fourthterm  = np.log(pwi)
    g = firstterm - secondterm - thirdterm + fourthterm
    return g

    #############################
    ## HW 1: 2.4 Mahlanobis Dist
    #############################
def get_mahalanobis_dist(x, mu, sigma):
    dist = np.dot(np.transpose(x - mu), np.dot(np.linalg.inv(sigma), (x - mu)))
    return dist

def get_mean_cov(dataset):
    #data = dataset[:, 0:2]
    mean = np.mean(dataset, axis=0)
    cov  = np.cov(dataset.T)
    return mean, cov

def naive_bayes(traindata, mean, pw, testdata):
    predictions = []
    for point in newdata:
        max_likelihood = []
        for idx, men in enumerate(mean):
            # multiply the probability of p(w1) for this class and datapoint
            # get_probability_score
            std0 = np.std(traindata[idx][:,0], axis=0)
            std1 = np.std(traindata[idx][:,1], axis=0)
            g1 = get_probability_score(point[0], men[0], std0, 1)
            g2 = get_probability_score(point[1], men[1], std1, 1)
            p = g1 * g2 * pw[idx]
            max_likelihood.append([p])
        max_l = np.argmax(max_likelihood)
        predictions.append(max_l)
    return predictions

# HW 3.1
def freidman_test(x):
    '''
    N       = len(dataset) # rows = datasets
    k       = len(dataset[0]) # cols = classifiers
    ranks = []
    # gets rank of each classifier per dataset
    for row in range(N): 
        ranks.append(np.argsort(dataset[k]))

    rankmat = np.array(ranks)
    #print(rankmat.shape)
    result = []
    for alg in range(k):
        Rj      = 1/N * np.sum(rankmat[:, alg])
        xf      = 12*N / (k*(k+1)) *  (np.sum(Rj^2) - k(k+1)^2 / 4)
        ff      = (N-1) * xf / ((N * (k-1)) - xf^2)
        result.append([ff])
    '''
    # using the scipy stats friedmanchisquare
    result = stats.friedmanchisquare(x[:,0], x[:,1], x[:,2], x[:,3], x[:,4], x[:,5])
    return result

def sample(M, Nints, prob):
    return np.random.choice(Nints, size=(M,1), p=prob)

if __name__ == "__main__":

    ######################
    ## HW 1: 2.1
    ######################
    # means for the 3 classes
    means = [[5., 6.], [2., 2.], [1., 7.]]
    # covariance matrices for the 3 classes
    covs = [[[ 1, 1],
            [1,  2]],
            [[ 1,  .5],
            [ 2,  0]],
            [[ .5,  0],
            [ 1,  .5]]]

    dim      = 2
    classes  = 3
    nsamples = 50
    dataset  = np.zeros(( nsamples*classes, dim + 1))
    for idx, mean in enumerate(means):
        xy  = make_random_var(mean, covs[idx], 2, nsamples)
        z   = np.full(xy.shape[0], idx)
        dataset[nsamples*idx:nsamples*(idx+1)] = np.array([xy[:, 0], xy[:, 1], z]).T

    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colormap = np.array(['r', 'g', 'b'])
    for x,y,lab in zip(dataset[:,0],dataset[:,1],dataset[:,2]):
            ax.scatter(x,y, c =colormap[int(lab)])
    plt.show()
    '''

    #############################
    ## HW 1: 2.3
    #############################
    # with the two classes, for each class, find mean and cov, then take each dataset and
    # find the descriminant

    mean = []
    cov  = []
    dat  = []
    pw   = []
    #des  = []
    total_data = len(dataset)
    for i in range(classes):
        a           = np.array([x for x in dataset if x[ 2] == i])
        a           = a[:, 0:2]
        amean, acov = get_mean_cov(a)
        pwi         = float(len(a))/total_data
        #g           = get_descriminant(a, amean, acov, len(a), pwi)
        mean.append(amean)
        cov.append(acov)
        dat.append(a)
        pw.append(pwi)
        #des.append(g)
    #print(g[0])

    #generate new dataset and compare it using g
    nmeans = [1., 0.]
    # covariance matrices for the 3 classes
    ncovs = [[ 6, 1],
             [1,  7]]
    
    #newdata = make_random_var(nmeans, ncovs, 2, 10)

    newdata = np.array([[0,1], [3,3], [2,5], [7,7], [12,1],[7,0], [3,5], [1,10], [0,0], [9,5]] )

    n = []
    for p in range(len(mean)):
        h = []
        for date in newdata:
            g = get_descriminant(date, mean[p], cov[p], len(date), pw[p])
            h = h + [g]
        n = n + [h]
    gscores = np.array(n)
    classification = np.argmax(gscores, axis=1)
    print("The classification for the generated 10 points is: {}".format(classification+1))



    #############################
    ## HW 1: 2.4
    #############################
    # find the mahalanobis dist between x and some mean vector given sigma
    nmeans, ncovs = get_mean_cov(newdata)
    distance = []
    for date in newdata:
        mal = get_mahalanobis_dist(date, nmeans, ncovs)
        distance.append(mal)
    print('Malanobis distance from those (new) points to their calculated mean is {}'.format(distance))


    #############################
    ## HW 1: 2.5 Naive Bayes Classifier
    #############################
    predictions = naive_bayes(dat, mean, pw, newdata)

    
    # compare to sklearn naive bayes
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(dataset[:, 0:2],dataset[:,2] ).predict(newdata)

    print('The predictions for the 10 new datapoints are (mine): {}'.format(predictions))
    print('The predictions for the 10 new datapoints are (sklearn): {}'.format(y_pred))
    print('Number of differing points between the sklearn and my naive bayes is {} out of {}'.format(( predictions != y_pred).sum(), len(newdata)))
    

    #############################
    ## HW 1: 3.1 Comparing Classifiers
    #############################
    # read in the file
    with open('hw1-scores.txt') as f:
        scores = f.readlines()
    # convert it to a list of scores for each line
    processedlines = []    
    for score in scores:    
        score = score.strip()
        scorelist = score.split(',')
        line = [float(x) for x in scorelist]
        processedlines.append(line)

    #print(processedlines[0])
    scorecard = np.array(processedlines)

    freid = freidman_test(scorecard)
    print(freid)
    print('Therefore we can say that the null hypothesis:')
    print('That the mean for each item is equivalent is false using p=.05')
    q = np.argmax(np.argmax(scorecard,axis=0))
    print ('The max error is found in the {}th classifier'.format(q+1))
    print('The mean error for each of the classifiers (6) is: ')
    print(np.mean(scorecard, axis=0))
    #print('The mean of the cov for each of the classifiers (6) is: ')
    #print(np.mean(np.cov(scorecard.T), axis=0))
    err = np.argmin(np.mean(scorecard,axis=0))
    print('And the least error is found in the {}nd class'.format(err+1) )


    #############################
    ## HW 1: 3.2 Sampling from Distribution
    #############################
    M       = 10
    Nints   = np.arange(1,M)
    prob    = [0.1, 0.05, 0.05, 0.1, 0.1,  0.2, 0.1, 0.1, 0.2]
    # to get a random probability
    # prob = np.random.random(M)
    # prob /= prob.sum()
    M       = 10000
    setN    = sample(M, Nints, prob)

    listN   = np.ndarray.tolist(setN)
    perc    = []
    for n in Nints:
        k = listN.count(n)
        perc.append(k/M)
    
    diff = abs(np.array(prob) - np.array(perc))
    print(diff)

