"""
Author: Neev Bhandari 

Inspired from the human brain, artificial neural networks (ANNs) are a 
type of computer vision model to classify images into certain categories.
In particular, in this project we will consider ANNs for the tasks of
recognising handwritten digits (0 to 9) from black-and-white images with a
resolution of 28x28 pixels.
"""

# Part 1
def linear(x, w, b): 
    """Provides output of single output node of the ANN

    Input: A list of inputs (x), a list of weights (w) and a bias (b).
    Output: A single number corresponding to the value of f(x) in Equation 1.

    Example:
    >>> x = [1.0, 3.5]
    >>> w = [3.8, 1.5]
    >>> b = -1.7
    >>> round(linear(x, w, b),6) #linear(x, w, b)
    7.35

    The problem involves taking an input of a node which involves 2 incoming edges and
    computing its output by multiplying the input by the weight of the edges and adding
    the bias attached to the node. It is important that the values of both inputs are
    calculated and their sum is returned.

    I chose to do this with a for loop in order to iterate over each input and multiplying
    it by the weight and adding the bias before adding the result to an output variable
    previously initialised to 0.
    """
    output = 0
    for i in range(len(x)):
        output += x[i] * w[i]
    output += b
    return output


def linear_layer(x, w, b): 
    """Computes output of the final layer of the ANN

    Input: A list of inputs (x), a table of weights (w) and a list of 
           biases (b).
    Output: A list of numbers corresponding to the values of f(x) in
            Equation 2.
    
    Example:
    >>> x = [1.0, 3.5]
    >>> w = [[3.8, 1.5], [-1.2, 1.1]]
    >>> b = [-1.7, 2.5]
    >>> y = linear_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #linear_layer(x, w, b)
    [7.35, 5.15]

    The problem involves finding the output of an ANN layer of two nodes as
    opposed to a single node, therefore it is important to account for
    multiple biases and sets of weights as well as which node they belong to in
    order produce the correct output. I tried to solve the problem by saving values
    which would ensure the correct weights and biases were used for their corresponding
    inputs.

    In my implementation, I used a nested for loop in order to iterate through the weights
    since they are contained in lists within the list. I also created a valiable,
    bias_counter which increments by 1, in order to update the bias so that it corresponds
    to the node.
    """
    output = []
    bias_counter = 0
    for i in w:
        answer = 0
        for j in range(len(x)):
            answer += (x[j] * i[j])
        answer += b[bias_counter]
        output.append(answer)
        bias_counter += 1
    return output


def inner_layer(x, w, b): 
    """Computes output of a single inner layer of the ANN

    Input: A list of inputs (x), a table of weights (w) and a 
           list of biases (b).
    Output: A list of numbers corresponding to the values of f(x) in 
            Equation 4.

    Example:
    >>> x = [1, 0]
    >>> w = [[2.1, -3.1], [-0.7, 4.1]]
    >>> b = [-1.1, 4.2]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [1.0, 3.5]
    >>> x = [0, 1]
    >>> y = inner_layer(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inner_layer(x, w, b)
    [0.0, 8.3]

    The problem is similar to the computation of an outer layer but it also involves an
    extra condition which involves outputting 0.0 if the initial output is greater than
    or equal to zero.

    For my implementation, I used the linear function in order to calculate the output
    and then added a conditional which would test if the output is greater than 0 and
    append 0.0 to the output variable if not.
    """
    output = []
    for i in range(len(b)):
        answer = linear(x, w[i], b[i])
        if answer > 0:
            output.append(answer)
        else:
            output.append(0.0)
    return output





def inference(x, w, b): 
    """Computes the output of a complete ANN

    Input: A list of inputs (x), a list of tables of weights (w) and a table
           of biases (b).
    Output: A list of numbers corresponding to output of the ANN.
    
    Example:
    >>> x = [1, 0]
    >>> w = [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> b = [[-1.1, 4.2], [-1.7, 2.5]]
    >>> y = inference(x, w, b)
    >>> [round(y_i,6) for y_i in y] #inference(x, w, b)
    [7.35, 5.15]

    This problem needed to calculate the output of the entire ANN which meant it needed to
    feed the output of one layer as the input of the next and the function also needed to vary
    slightly for the final layer since it outputs what it computes regardless of whether it is
    greater than zero unlike the inner layers.

    I used a list to store the input for the current layer to which I appended the output of
    each layer so that it would update the input used as it goes to the next layer. I used a
    range sequence to make a for loop that would iterate over all but the last layer and append
    the result of the function for the inner layer. Finally I returned the output of the final
    layer function on the final element of my inputs list.
    """
    inputs = [x]
    for i in range(len(w)-1):
        inputs.append(inner_layer(inputs[i], w[i], b[i]))
    return linear_layer(inputs[-1], w[-1], b[-1])

def weights_to_float(x):
    """Converts list in list in list representing weights of edges going into node in ANN

    input: List in list in list containing strings representing weights of edges of ANN

    output: List in list in list of floats representing weights of edges of ANN

    Example:
    >>> w = weights_to_float([[['2.1', '-3.1'], ['-0.7', '4.1']], [['3.8', '1.5'], ['-1.2', '1.1']]])
    >>> w
    [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    """
    for i in x:
        for j in i:
            for k in range(0,len(j)):
                j[k] = float(j[k])
    return x

def read_weights(file_name): 
    """Converts strings of weight values from a file to tables of weights of edges in ANN

    Input: A string (file_name) that corresponds to the name of the file
           that contains the weights of the ANN.
    Output: A list of tables of numbers corresponding to the weights of
            the ANN.
    
    Example:
    >>> w_example = read_weights('example_weights.txt')
    >>> w_example
    [[[2.1, -3.1], [-0.7, 4.1]], [[3.8, 1.5], [-1.2, 1.1]]]
    >>> w = read_weights('weights.txt')
    >>> len(w)
    3
    >>> len(w[2])
    10
    >>> len(w[2][0])
    16

    The function reads a file, splits it at ',' and strip empty spaces. The function then
    needs to sort the weights into seperate lists which can be used by the ANN. The function
    also needs to account for # and discard of them while also using a new string.

    I used a for loop to iterate through each line in the file, strip spaces and split at
    commas. Then I used the remove method to remove the first # before iterating through
    the rest of the weights. I then used another for loop and if statement to either
    add the element to the list or add the list to the output and create a new list
    if a # is encountered. Finally, to convert each individual weight to a float,
    I called the weights_to_float function.
    """
    weights = []
    f = open(file_name)
    for l in f:
        l = l.strip()
        l = l.split(',')
        weights.append(l) 
    weights_final = []
    weights.remove(['#'])
    new = []
    for w in weights:
        if w != ['#']:
            new.append(w)
        else:
            weights_final.append(new)
            new = []
    weights_final.append(new)
    weights_to_float(weights_final)
    return weights_final


def biases_to_float(x):
    """Coverts bias values of nodes in ANN in a table to floats

    input: List in list containing strings representing biases of nodes of ANN

    output: List in list of floats representing biases of nodes of ANN

    Example:
    >>> b = biases_to_float([['4.6', '3.0'], ['1.2', '7.1']])
    >>> b
    [[4.6, 3.0], [1.2, 7.1]]
    """
    for i in x:
        for j in range(0,len(i)):
            i[j] = float(i[j])


def read_biases(file_name): 
    """Converts strings of biases from a file to table containing biases of nodes in ANN

    Input: A string (file_name), that corresponds to the name of the file
           that contains the biases of the ANN.
    Output: A table of numbers corresponding to the biases of the ANN.
    
    Example:
    >>> b_example = read_biases('example_biases.txt')
    >>> b_example
    [[-1.1, 4.2], [-1.7, 2.5]]
    >>> b = read_biases('biases.txt')
    >>> len(b)
    3
    >>> len(b[0])
    16

    This function is similar to the reading_weights function, so the code is similar
    to the reading_weights function, however, the biases are only stored in a list in a list
    as opposed to weights which are are a list in a list in a list since one node will only have 
    one bias but multiple incoming edges, so the function needed fewer for loops.

    Similarly to the function reading_weights, I first stripped and split the file
    before removing the first '#' and then appending biases to a list until a '#'
    at which point I would append my new list to the list of all biases before repeating
    the process. The biases_to_floats function only looks into the lists contained in the
    biases_final list and converts them to floats.
    """
    biases = []
    f = open(file_name)
    for l in f:
        l = l.strip()
        l = l.split(',')
        biases.append(l) 
    biases_final = []
    biases.remove(['#'])
    for b in biases:
        if b != ['#']:
            biases_final.append(b)   
    biases_to_float(biases_final)
    return biases_final


def read_image(file_name): 
    """Converts string in file to list of numbers representing input of ANN
    
    Input: A string (file_name), that corresponds to the name of the file
           that contains the image.
    Output: A list of numbers corresponding to input of the ANN.

    Example:
    >>> x = read_image('image.txt')
    >>> len(x)
    784

    The image file needs to be converted into a list so it can be used in the inference
    function for the ANN. The image is presented in 28 lines so it was important to make
    sure only characters were included and the new line was not counted. The elements of
    the string then needed to be converted into integers and added to a list so they could
    be read my the inference function.

    I used the strip list method to remove the '\n' at the end of each line so it would not
    be counted. I then used a for loop to iterate through the index of each character in
    the string to convert it into an index before using the append list method to add it
    to the output.
    """
    output = []
    f = open(file_name)
    for line in f:
        new = []
        line = line.strip('\n')
        for i in range(len(line)):
            output.append(int(line[i]))
    return output


def argmax(x):
    """Finds maximum of list pertaining to scores of each possible digit

    Input: A list of numbers (i.e., x) that can represent the scores 
           computed by the ANN.
    Output: A number representing the index of an element with the maximum
            value, the function should return the minimum index.
    
    Example:
    >>> x = [1.3, -1.52, 3.9, 0.1, 3.9]
    >>> argmax(x)
    2

    The problem was quite simple as it just involved iterating through a list of scores,
    finding the greatest and then outputting the corresponding digit.

    I used the max in-built function in order to determine the highest score before using
    the index list method to return the value of the digit since it is the same as its
    list index
    """
    m = max(x)
    output = x.index(m)
    return output


def predict_number(image_file_name, weights_file_name, biases_file_name): 
    """Takes image of handwriting and returns digit which it most likely represents

    Input: A string (i.e., image_file_name) that corresponds to the image
           file name, a string (i.e., weights_file_name) that corresponds
           to the weights file name and a string (i.e., biases_file_name)
           that corresponds to the biases file name.
    Output: The number predicted in the image by the ANN.

    Example:
    >>> i = predict_number('image.txt', 'weights.txt', 'biases.txt')
    >>> print('The image is number ' + str(i))
    The image is number 4

    The problem was putting everything together to take the input of a file with a string
    corresponding to the image which is then put through an ANN to decipher which digit from
    0-9 it is most likely to be. For this the best method was using all the functions I had
    previously defined and putting them in order to process the image and figure out which
    digit the image corresponds to.

    I used previous functions in order to read the files of inputs, weights and biases and
    assigned the outputs to variables which I then used as the inputs for the inference
    function. I then used the output of the inference function and found the most likely digit
    through the argmax function which I output as the digit corresponding to the handwriting.
    """
    x = read_image(image_file_name)
    w = read_weights(weights_file_name)
    b = read_biases(biases_file_name)
    probabilities =inference(x, w, b)
    output = argmax(probabilities)
    return output 


def flip_pixel(x):
    """Flips value of pixel from 0 to 1 or vice versa

    Input: An integer (x) representing a pixel in the image.

    Output: An integer representing the flipped pixel.

    Example:
    >>> x = 1
    >>> flip_pixel(x)
    0
    >>> x = 0
    >>> flip_pixel(x)
    1
    
    The problem was fairly simple as it just requires me to return a 0 when the input is 1
    and a 1 when the input is 0. I just needed to check what the input is and then return
    the other integer.

    I used an if and elif statement which would dictate what the function does based on the
    input
    
    """
    if x == 0:
        return 1
    elif x == 1:
        return 0

def modified_list(i,x):
    """flips specified pixel in given list

    Input: A list of integers (x) representing the image and an integer (i) representing the
    position (i.e., index) of the pixel.

    Output: A list of integers (x) representing the modified image.

    Example:
    >>> x = [1, 0, 1, 1, 0, 0, 0]
    >>> i = 2
    >>> modified_list(i,x)
    [1, 0, 0, 1, 0, 0, 0]


    This problem was not too hard either as I just needed to execute flip pixel on a given element in a list.
    So I just needed to reassign the specified element to the opposite integer.

    For this, I simply assigned the element at the given index in the list and called on the flip_pixel function
    which allowed me to flip the pixel at the specified index.
    
    """
    x[i] = flip_pixel(x[i])
    return x

def compute_difference(x1,x2):
    """Finds the numbr of differences between two lists

    Input: A list of integers (x1) representing the input image and a list of integers (x2) representing the adversarial image.

    Output: An integer representing the total absolute difference between the elements of x1 and x2.

    Example:
    >>> x1 = [1, 0, 1, 1, 0, 0, 0]
    >>> x2 = [1, 1, 1, 0, 0, 0, 1]
    >>> compute_difference(x1,x2)
    3

    For this problem, I needed to compare each element of the two lists at the same index and count how many are
    different between them.

    I used a for loop to iterate over the indices of the list and then put an if statement inside which compared the
    elements of each list at the index. If they are different, I incremented the previously initialised count
    variable by 1, which was returned in the end.
    """
    count = 0
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            count += 1
    return count

def select_pixel(x, w, b):
    """Finds the pixel in the image which has the maximum impact when flipped

    Input: A list of inputs (x), a list of tables of weights (w) and a table of biases (b).

    Output: An integer (i) either representing the pixel that is selected to be flipped, or with value -1 representing
    no further modifications can be made.

    Example:
    >>> x = read_image('image.txt')
    >>> w = read_weights('weights.txt')
    >>> b = read_biases('biases.txt')
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    238
    
    >>> x = modified_list(pixel,x)
    >>> pixel = select_pixel(x, w, b)
    >>> pixel
    210

    This problem required a lot more work as I needed to find a way to test the image with each pixel flipped and calculate the impact. This was a bit tricky but in the end I decided to make the original score of the first positive and the score I was testing negative since the score of the most likely number goes down but for the score of the second most likely number, I did it the opposite way as it goes up. This way, I was able to add these values up to find the pixel to be flipped which has the highest impact. I kept comparing this to the best possible pixel I had found so far and reassigned the variable if I found a better pixel to flip.

    First, I calculate the scores of each digit calling my inference function before copying the list y slicing so that I can sort the values while keeping the original list intact. I then saved the top two scores in the sorted list as the variables 'first' and 'second' respectively and then called the index list method to find the corresponding indices in the original list to get the first and second most likely digits. After initialising values for the biggest impact and its corresponding flipped pixel, I used a for Loop to iterate over the length of the image file. I then used slicing again to modify the pixel at the corresponding i value before finding the scores of the 'first' and 'second' values. I then calculated the impact created by this modified image and if it is greater than the current highest impact I have stored, I reassign the best impact and its corresponding pixel. If no changes are made, the variable storing the best pixel to flip will remain unchanged and therefore the function returns -1, otherwise the best pixel to flip is returned.
    """
    probabilities = inference(x, w, b)
    sorted_probabilities = probabilities[:]
    sorted_probabilities = sorted(sorted_probabilities)   
    first_prob = sorted_probabilities[-1]
    second_prob = sorted_probabilities[-2]  
    first = probabilities.index(first_prob)
    second = probabilities.index(second_prob)  
    pixel = 0
    best_impact = 0    
    for i in range(len(x)):
        new_x = modified_list(i,x[:])
        test_probabilities = inference(new_x, w, b)
        test_first_prob = test_probabilities[first]
        test_second_prob = test_probabilities[second]
        impact = first_prob - test_first_prob + test_second_prob - second_prob
        if impact > best_impact:
            best_impact = impact
            pixel = i
    if pixel == 0:
        return - 1
    else:
        return pixel


def write_image(x, file_name):
    """writes a list x into a file as a 28x28 pixel image

    Input: A list of integers (x) representing the image and a string (file name) representing the file name. 
    
    Output: Write out each pixel represented in the list x to a file with the name file name as a 28x28 image.
    
    Example:
    >>> x = read_image('image.txt')
    >>> x = modified_list(238,x)
    >>> x = modified_list(210,x)
    >>> write_image(x,'new_image.txt')

    This function was not too hard but it was quite technical and I needed to make sure to open the file in wrte mode as well as 
    create a new line after every 28 elements in order to produce a 28x28 image.

    I used a variable 'file' and assigned it to the file name which was opened in write mode. I then initialised a variable count,
    which allowed me to iterate over the elements of the list and terminate my while loop when the whole list has been covered.
    Then with a nested for loop, I iterated over each element while also creating a new line after every 28 elements to create a 28x28
    pixel image.
    """
    file = open(file_name, 'w')
    count = 0
    while count != 784:
        for i in range(28):
            for j in range(28):
                file.write(str(x[count]))
                count += 1
            file.write('\n')


def adversarial_image(image_file_name,weights_file_name,biases_file_name):
    """Generates adversarial image

    Input: A string (i.e., image file name) that corresponds to the image file name, a string (i.e., weights file name) that corresponds to the weights file name and a string (i.e., biases file name) that corresponds to the biases file name.
    
    Output: A list of integers representing the adversarial image or the list [-1] if the algorithm is unsuccesful in finding an adversarial image.

    Example:
    >>> x1 = read_image('image.txt')
    >>> x2 = adversarial_image('image.txt','weights.txt','biases.txt')
    >>> if x2[0] == -1:
    ...    print('Algorithm failed.')
    ... else:
    ...    write_image(x2,'new_image')
    ...    q = compute_difference(x1,x2)
    ...    print('An adversarial image is found! Total of ' + str(q) + ' pixels were flipped.')
    ...
    An adversarial image is found! Total of 2 pixels were flipped.

    This function was quite difficult as there were many parts to it mand many things needed to be stored including the pixel that
    is flipped and the new image produced. Eventually I realised that the function keeps flipping a certain pixel after a point
    which means that is has no more adjustments to make and this is how I came to the realisation that I need to terminate my while
    loop when this is the case and return the second last image stored in my list.

    I first created a list to store the pixels that are flipped and added the first pixel to be changed by calling my select_pixel 
    function. Then, I chacked to see if it had returned -1, meaning no further adjustments could be made in which case, I return 
    pixel_lis which is [-1], otherwise, the function proceeds to add the image with the pixel modified into image_list. In order
    to make my while loop work, I had to repeat this process once again so that the condition for my while loop could be tested but
    I also had to test whether the same pixel had been flipped twice in case there was only one pixel to be flipped. If not, the 
    function enters the while loop which executes until the same pixel is flipped twice. Within the while loop, I call select pixel
    on the most image and add the new modified image to the list until the loop terminates. When the loop is exited, the second
    last item of image_list is output as the last one is simply the same pixel flipped back since there are no more changes to make.
    """
    x = read_image(image_file_name)
    w = read_weights(weights_file_name)
    b = read_biases(biases_file_name)
    pixel_list = [select_pixel(x, w, b)]
    if pixel_list[0] == -1:
        return pixel_list
    else:
        image_list = [modified_list(pixel_list[-1], x)]
        pixel_list.append(select_pixel(image_list[-1], w, b))
        if pixel_list[0] == pixel_list[1]:
            return image_list[0]
        else:
            image_list.append(modified_list(pixel_list[-1], x))
    
    while not select_pixel(image_list[-1], w, b) == select_pixel(image_list[-2], w, b):
        pixel_list.append(select_pixel(image_list[-1], w, b))
        image_list.append(modified_list(pixel_list[-1], image_list[-1]))
    return image_list[-2]
                   

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
    

