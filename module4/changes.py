def convergence_reached(self):
        if self.problem.get_output_neuron_structure() == "2D_lattice":
            pass     # Todo

        elif self.problem.get_output_neuron_structure() == "ring":
          
            for neuron in self.output_neurons:
                if len(neuron.attached) == 0:   # if there is not a one-to-one relationship between input and output nodes
                    return False

            # Check distance between 
            for neuron in self.input_neurons:
                if euclidian_distance(neuron, neuron.get_output_neuron) > self.legal_radius:
                    return False
            
            # Todo: legg til flere krav til convergence.
            return True



def compute_total_cost(self):
        if self.problem.get_output_neuron_structure() == "2D_lattice":
            pass     # Todo

        elif self.problem.get_output_neuron_structure() == "ring":
            # Cost is equal to dist(xN, x0) + sum(dist(xi, xi+1) for i in output_neurons)
            return euclidian_distance(self.output_neurons[0], self.output_neurons[-1]) + sum([euclidian_distance(x, self.output_neurons[i+1]) for i,x in enumerate(self.output_neurons[:-1])])
        return 0