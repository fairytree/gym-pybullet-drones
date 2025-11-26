import numpy as np
from gymnasium import spaces

class ParticleFilter:
    def __init__(self,
                 N=1000,
                 bounds=None,
                 sigma=0.1,
                 std_dev=0.01,
                 seed=None):
        self.rng=np.random.default_rng(seed)
        self.N=int(N)
        self.sigma=float(sigma)
        self.std_dev=float(std_dev)

        #ensure there are bounds given, then make into (3,2) for x,y,z upper/lower
        if bounds is None:
            raise ValueError("In Particle Filter: missing bounds")
        else:
            self.bounds=np.asarray(bounds, dtype=float).reshape(3,2)
        self.particles = self._uniform_sample(self.N)
        self.weights=np.ones(self.N).self.N
    
    ##Creates a uniform sample accross the bounds
    def _uniform_sample(self, N):
        low=self.bounds[:,0]
        high=self.bounds[:,1]
        samples=self.rng.uniform(low=low, high=high, size=(N,3))

    def predict(self):
        #confirm there is a deviation, if so add noise
        if self.std_dev > 0:
            noise=self.rng.normal(scale=self.std_dev, size=(self.N,3))
            self.particles += noise

    def _prob(self, sensor_pos, sensor_read):
        """
        sensor_pos: location of sensor/drone
        sensor_read: new reading from sensor
        """
        # predict range of each particle
        delta=self.particles=sensor_pos.reshape((1,3))
        ranges = np.linalg.norm(delta, axis=1)

        residual=(sensor_read-ranges)/self.sigma

        prob=np.exp(-0.5*(residual**2))

        return prob
    
    def update(self, sensor_pos, sensor_read, resample=False, resample_thresh=None):
        """
        sensor_pos: where is the sensor
        sensor_read: what is the new reading
        resample: should we resample if sample size is too small
        resample_threshold: how small to trigger resample
        """
        prob=self._prob(sensor_pos,sensor_read)

        weights=self.weights*prob

        sum_weights=weights.sum()

        # normalizing probabilities
        # if the probabilities are all very low, reset weights
        if sum_weights <=0 or not np.isfinite(sum_weights):
            weights=np.ones(self.N)/self.N
        else:
            weights=weights/sum_weights
        
        self.weights=weights

        if resample:
            ess = self.effective_sample_size()
            if resample_thresh is None:
                #default threshold: N/2
                ess_thresh=max(1, self.N/2)
            if ess < ess_thresh:
                self.resample()
    
    def effective_sample_size(self):
        return 1.0/np.sum(self.weights **2)
    
    def resample(self):
        """
        Resample, leaving unifrom weights after sampling
        """

        N=self.N
        positions=(self.rng.random()+np.arrange(N))/N
        cum_weights=np.cumsum(self.weights)
        new_particles=np.empty_like(self.particles)
        i,j=0,0
        while i<N and j<N:
            if positions[1]<cum_weights[j]:
                new_particles[i]=self.particles[j]
                i += 1
            else:
                j += 1
        self.particles = new_particles
        self.weights.fill(1.0/N)
    
    def estimate_mean_covariance(self):
        """
        Returns posterior mean and covariance
        uses weighted mean and covariance of particles
        """
        mean=np.sum(self.particles * self.weights.reshape(-1,1), axis=0)
        differences = self.particles - mean.reshape((1,3))

        #weighted covariance
        covariance = (differences.T*self.weights)@differences

        #ensure symmetry
        covariance = 0.5*(covariance + covariance.T)
        return mean, covariance
    
    def entropy_approximation(self):
        """
        Approximate differential entropy
        """

        # Shannon entropy of weights: -sum weights log weights
        weights = np.clip(self.weights, 1e-12, 1.0)
        entropy=-np.sum(weights*np.log(weights))
        return entropy
    
    def get_top_k_modes(self, k=3, min_seperation=0.1):
        """
        Returns the k highest-weight particles, enforcing minimum separation to avoid duplicates
        """
        idx = np.argsort(self.weights)[::-1]
        modes = []
        for i in idx:
            p=self.particles[i]
            if all(np.linalg.norm(p-m)>=min_seperation for m in modes):
                modes.append(p.copy())
                if len(modes)>=k:
                    break
        return np.array(modes)
    
    def reinitialize_uniform(self):
        self.particles = self._uniform_sample(self.N)
        self.weights.fill(1.0/self.N)

    def KL_divergence(self, sensor_pos, sensor_read):
        prior_weights=self.weights.copy()

        self.update(sensor_pos,sensor_read, resample=True)
        post_weights=self.weights.copy()
        KL=np.sum(post_weights*(np.log(post_weights+1e-12)-np.log(prior_weights+1e-12)))

        return KL