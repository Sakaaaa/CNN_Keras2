function y=to_categorical(y, num_classes)
    %Converts a class vector (integers) to binary class matrix.
    %E.g. for use with categorical_crossentropy.
    % Arguments
    %    y: class vector to be converted into a matrix
    %        (integers from 0 to num_classes).
    %    num_classes: total number of classes.
    % Returns
    %    A binary matrix representation of the input.
    
    [n,~] = size(y);
    categorical = zeros(n, num_classes);
    for i=1:n
        categorical(i,y(i)) = 1;
    end
    y=categorical;