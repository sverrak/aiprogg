
def get_input():
    # Until told, the algorithm should run infinitely
    while True:
        mode = input("Do you want to type all parameters: ")
        if(mode=="yes"):

            # Choose dataset
            print("Candidate datasets: 'mnist', 'wine', 'glass', 'yeast' ")
            dataset = input("Choose dataset: ")
            
            # Get input values
            method = 0
            method = input("What method do you want to use (GD, ... ): ")
            lr = float(input("Learning rate: "))
            n_hidden_layers = int(input("Hidden layers: "))
            hidden_layers = []
            activation_functions = []
            for i in range(n_hidden_layers):
                print("\nParameters for hidden layer " + str(i) + ".")
                
                # Collecting the inputs
                sizeX = float(input("Layer size: "))
                hidden_layers.append([sizeX])
            #activation_functions.append(input("Input layer activation function:" ))
            activation_functions.append(input("Hidden layer activation function (relu, softmax, sigmoid, tanh):" ))
            activation_functions.append(input("Output layer activation function (relu, softmax, sigmoid, tanh):" ))
            cost_function = input("Cost function (ce, mse, ..): ")

            case_fraction = float(input("Case fraction: "))
            vfrac = float(input("Validation fraction: "))
            tfrac = float(input("Test fraction: "))
            wr0 = int(input("Lower weight range: "))
            wr1 = int(input("Upper weight range: "))
            mbs = int(input("MBS: "))
            wrange = [wr0, wr1]
        else:
            dataset = "mnist"
            
            # Get input values
            method = 0
            method = input("What method do you want to use (GD, ... ): ")
            lr = 0.05
            hidden_layers = [9,10,3]
            activation_functions = ["relu","softmax"]
            cost_function = "MSE"

            case_fraction = 1.0
            vfrac=0.1
            tfrac=0.1
            wr0 = -0.1
            wr1 = 0.1
            mbs = None
            wrange = [wr0, wr1]

            epochs=300
            nbits=4
            
            showint=100
            
            
            vint=100
            sm=False

        # Run the GANN
        print("Computing optimal weights....")
        

        gann_runner(dataset, method, lrate, hidden_layers, activation_functions[0], activation_functions[1], cost_function, case_fraction, vfrac, tfrac, wrange, mbs)

        dims = [input_size] + hidden_layers + [output_size]
        

        print("Done computing weights!\n")

    


get_input()