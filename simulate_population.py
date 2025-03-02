import matplotlib.pyplot as plt
import numpy as np
import scipy.special as special
#This file contains the function Simulate which
# simulates a population under selection, mutation, genetic drift.



def generate_mutated(size,thetaN):
	"""
Parameters
----------
size : the size
thetaN: the mutation rate. The probability of mutation from -1 to +1
       is thetaN[0]/N
	"""
	optimum_mutation = thetaN[0]/(thetaN[0]+thetaN[1])
	return np.random.choice([True,False],
				size=size,
				p=(optimum_mutation,1-optimum_mutation)).astype("bool")

def generate_pop(thetaN,N,L):
	"""Generates a population with allele density
	 x**(2thetaN[0]-1)(1-x)**(2thetaN[1] - 1)"""
	x = np.linspace(1/N,1-1/N,N)
	equilibrium_distribution = x**(2*thetaN[0]-1) * (1-x)**(2*thetaN[1]-1)
	equilibrium_distribution /= np.sum(equilibrium_distribution)
	frequencies = np.random.choice(x,p=equilibrium_distribution,size=L)
	population = np.array([[[False]*2]*L]*N,dtype="bool")
	for l in range(L):
		proba = (frequencies[l],1-frequencies[l])
		population[:,l,:] = np.random.choice([True,False],p=proba,size=(N,2))
	return population

def equaleffects(L):
	"""When all allelic effects are the same we use this function as alphamethod"""
	return np.ones(L)/L

def random_exponential(L):
	"""When"""
	return np.random.exponential(1/L,size=L)

class Population():
	"""A population has
        population : a (N,L) matrix of bools representing the L genes of each of the N organisms
	meantrait: mean trait value in the population
        vartrait: variance in trait value in the population
        alpha : a string of L floats representing the additive effects at each locus (picked at
                the beginning and then fixed)
	alphamethod: when alpha is None, the method to generate the additive effects as i.i.d variables
		(example: alphamethod = np.random.exponential). It has a single parameter L.
		 We recommand that alpha be of order 1/L when the mutation optimum eta is of order 1.
        """
	def __init__(self,thetaN,N,L,population=None,alpha=None,alphamethod=random_exponential):
		"""
                   Either supply alpha (a list of additive effects) or alphamethod (a function
                   to randomly generate the alphas)
                   """
		if alpha is None:
			self.alpha=alphamethod(L)
		else:
			self.alpha=alpha
		if population is None:
			population = generate_pop(thetaN,N,L)
		self.population = population
		self.meantraits = None
		self.vartraits = None

	def trait(self):
		return (self.population@np.ones(2)) @ self.alpha

	def fitness(self,omega,eta,L):
		dist_to_optimum = self.trait() - eta
		return np.exp(-1/(2*omega**2) * (dist_to_optimum)**2)

	def allele_frequencies(self):
		return np.mean(self.population,axis=(0,2))

	def mutation(self,thetaN,N,L):
		"""Mutation works by sampling a Poisson number of genes and
		   mutating them.
                   The mutation probability from -1 to +1 is thetaN[0]/N per
                   generation per locus per organism"""
		number_of_mutations=np.random.poisson(2*L*(thetaN[0]+thetaN[1]))
		mutated_indexes= (np.random.randint(N,size=number_of_mutations),
				 np.random.randint(L,size=number_of_mutations),
				 np.random.randint(2,size=number_of_mutations))
		self.population[mutated_indexes]=generate_mutated(number_of_mutations,thetaN)

	def selection_drift_sex(self,omega,eta,N,L):
		"""For each offspring we sample two parents proportionnally
		to fitness, and recombine their genomes"""
		population_fitnesses = self.fitness(omega,eta,L)
		sum_fitnesses = np.sum(population_fitnesses)
		population_fitnesses = population_fitnesses/sum_fitnesses

		list_parents = np.random.choice(N,p=population_fitnesses,size=2*N)
		list_crossing_overs = np.random.randint(L,size=2*N)
		list_chromosome_shuffling = np.random.randint(2,size=2*N)
		new_population = np.array([[[False]*2]*L]*N,dtype="bool")
		for n in range(N):
			parents = (list_parents[2*n],list_parents[2*n+1])

			new_genome = np.zeros((L,2),dtype="bool")

			crossing_over = list_crossing_overs[2*n:2*n+2]
			new_population[n,:,0] = np.append(
					self.population[parents[0],
							:crossing_over[0],
							list_chromosome_shuffling[2*n]],
					self.population[parents[0],
							crossing_over[0]:,
							1-list_chromosome_shuffling[2*n]]
					)
			new_population[n,:,1] = np.append(
					self.population[parents[0],
							:crossing_over[1],
							list_chromosome_shuffling[2*n+1]],
					self.population[parents[1],
							crossing_over[1]:,
							1-list_chromosome_shuffling[2*n+1]]
					)
		self.population = new_population
		self.meantraits = np.mean(self.trait())
		self.vartraits = np.var(self.trait())


def Simulate(thetaN,omega,eta,N,L,T,initial_pop=None,record_every = 1,alphamethod=random_exponential):
	"""Simulates population evolution over time T. Returns vector of allele frequencies.
	Parameters:
	----------
	thetaN: mutation rates. The probability of mutation from -1 to +1 is thetaN[0]/N per locus
               per generation per organism
	s
	N
	L
	T: optional.
	recond_every: optional:how often should the population be recorded ?

	Returns:
	list_allele_frequencies,list_varfitnesses,parameters
	"""
	nb_timesteps = int(T*N)+1
	list_meantraits = np.zeros(nb_timesteps//record_every)
	list_vartraits = np.zeros(nb_timesteps//record_every)
	pop = Population(thetaN,N,L,population=initial_pop,alphamethod=alphamethod)

	for tN,t in enumerate(np.linspace(0,T,nb_timesteps)):
		pop.selection_drift_sex(omega,eta,N,L)
		pop.mutation(thetaN,N,L)
		if tN%10==0:
			print("Done: step "+str(tN)+" of "+str(nb_timesteps))
		if tN%record_every==0:
			list_meantraits[tN//record_every] = pop.meantraits
			list_vartraits[tN//record_every] = pop.vartraits
	return pop.allele_frequencies(), pop.alpha, list_meantraits, list_vartraits
