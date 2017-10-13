def estimate_cost(current_state):
    if (SEARCH_MODE == "bfs" or SEARCH_MODE == "dfs"):
        return 0

    elif (SEARCH_MODE == "A*"):
        number_of_columns = len(current_state[0][0])
        number_of_rows = len(current_state) - number_of_columns
        min_cost = 0
        
        # If there are more than one row that have row domain size > 0, 
        # then there must be at least one column that is needed to reduce all column domains
        temp_counter_vertical = 0
        for i in range(number_of_rows):
            if(len(current_state[i]) > 1):
                temp_counter_vertical += 1
            if(temp_counter_vertical == 2):
                min_cost += 1
                break

        # Vice versa
        temp_counter_horizontal = 0
        for i in range(number_of_rows, number_of_columns+number_of_rows):
            if(len(current_state[i]) > 1):
                temp_counter_horizontal += 1
            if(temp_counter_horizontal == 2):
                min_cost += 1
                break

    
        # max {     1 if count(len(current_state[i]) > 1, i: row indices) > 1           else 0 
        #       +   1 if count(len(current_state[i]) > 1, i: column indices) > 1        else 0, 
        #           1 if count(len(current_state[i]) > 1, i: row+column indices) >= 1   else 0}
        
        # If there are at least two row values that are equal to 0 and two that are equal to 1, 
        # there must be yet another column that must be revised in order to decide which value that is correct
        # However, these tests are currently too slow.

        # Current best guess on potential improvements: 
        """
        for rc in current_state[:number_of_rows]:
            breaker = False
            if(len(rc)>0):
                for i in range(len(rc[0])):
                    count_positives = [0 for i in rc[0]]
                    count_negatives = [0 for i in rc[0]]
                    for rd in rc:
                        if (rd[i]==1):
                            count_positives[i] += 1
                        else:
                            count_negatives[i] += 1
                    if ((all(pos >= 2 for pos in count_positives)) and all(neg >= 2 for neg in count_negatives)):
                        min_cost += 1
                        breaker = True
                        break
            if(breaker == True):
                break

        # Vice versa
        for rc in current_state[number_of_rows:]:
            breaker = False
            if(len(rc)>0):
                for i in range(len(rc[0])):
                    count_positives = [0 for i in rc[0]]
                    count_negatives = [0 for i in rc[0]]
                    for rd in rc:
                        if (rd[i]==1):
                            count_positives[i] += 1
                        else:
                            count_negatives[i] += 1
                    if ((all(pos >= 2 for pos in count_positives)) or all(neg >= 2 for neg in count_negatives)):
                        min_cost += 1
                        breaker = True
                        break
            if(breaker == True):
                break
        """

        return max(min_cost, 1 if temp_counter_horizontal + temp_counter_vertical > 1 else 0)
        
        return min_cost
        



        # Old method that is not
        """product = math.exp(1)
        for i in range(len(current_state)):
            product = product * math.log(len(current_state[i])+1)

        if (product == 0):
            return 9999999999999999999999999999999999999999999999999999999999
        else:
            return product 
            return math.log(product+1)
            """

        # If A* is chosen, we compute the degree to which each column constraints is violated
