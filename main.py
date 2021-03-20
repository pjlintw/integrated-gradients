



def main():

	### Training ###
	if mode.train:
		train()

	### Evalutation ### 

def train():
	"""Adversarial training for CGANs.

	Args:
	  generator
	"""	
	for n_step in range(config.num_steps):
		
		# Sample positive examples

		# Sample noise examples

		# obtain generated data 
		generated_pred = generator()

		# Update discriminator
		reader.step()


		# Update generator
		generator.step()
		

def evaluationg():
	pass

if __name__ == "__main__":
	main()
