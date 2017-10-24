
# On line 315, call get unique values
labels, features = get_unique_values((labels, featuers))

# Insert in tutor3 / GANN
def get_unique_values(lisst):
    return list(set(lisst))

a = [("a","b"), ("b","b"), ("b","b")]
print(get_unique_values(a))

# Here I have added the possibility of using mean/stdev feature scaling
def scale_features(features, mode=1):
    # Max & min scaling
    if(mode==1):
        for c in range(len(features[0])):
            col_max = 0
            col_min = 9999999

            # Get the right min and max value for the column
            for f in features:
                if (f[c] > col_max):
                    col_max = f[c]
                elif (f[c] < col_min):
                    col_min = f[c]

            # Scale each feature value
            for f in features:
                f[c] = (f[c] - col_min) / (col_max - col_min)
    # Mean & stdev scaling
    else:
        for c in range(len(features[0])):
            col_my = sum([f[c] for f in features]) / len(features)
            col_sigma = np.std([f[c] for f in features])

            for f in features:
                f[c] = (f[c] - col_my) / col_sigma


    return features


# Help method for new do_training
def convergence_test(a, b):
        return math.abs(a-b) < 0.0001 # Insert code if this is requested


# Replace do_training in tutor3.py with this code
def do_training(self, sess, cases, epochs=100, continued=False):
    if not continued: self.error_history = []
    is_early_termination = input("Do you want to terminate early given convergence? ") == "yes"

    for i in range(epochs):
        if(is_early_termination and convergence_test(self.error_history[-1], self.error_history[-2])):
            print("Terminated run after " + str(i) + " epochs.")
            break 
        
        error = 0
        step = self.global_training_step + i
        gvars = [self.error] + self.grabvars
        mbs = self.minibatch_size
        ncases = len(cases)
        nmb = math.ceil(ncases / mbs)
        
        for cstart in range(0, ncases, mbs):  # Loop through cases, one minibatch at a time.
            cend = min(ncases, cstart + mbs)
            minibatch = cases[cstart:cend]
            inputs = [c[0] for c in minibatch]
            targets = [c[1] for c in minibatch]
            feeder = {self.input: inputs, self.target: targets}
            _, grabvals, _ = self.run_one_step([self.trainer], gvars, self.probes, session=sess,
                                               feed_dict=feeder, step=step, show_interval=self.show_interval)
            error += grabvals[0]
        self.error_history.append((step, error / nmb))
        self.consider_validation_testing(step, sess)


    self.global_training_step += epochs
    TFT.plot_training_history(self.error_history, self.validation_history, xtitle="Epoch", ytitle="Error",
                              title="", fig=not (continued))