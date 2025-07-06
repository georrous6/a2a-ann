function result = file_testing_function(matFile)
    % Load matrices from the provided .mat file
    data = load(matFile);
    
    % Ensure all required variables exist in the file
    if ~isfield(data, 'my_IDX') || ~isfield(data, 'my_D') || ...
       ~isfield(data, 'test_IDX') || ~isfield(data, 'test_D')
        disp('Error: Missing required matrices in the .mat file.');
        result = 1; % Return failure
        return;
    end
    
    % Extract matrices
    my_IDX = data.my_IDX;
    my_D = data.my_D;
    test_IDX = data.test_IDX;
    test_D = data.test_D;
    test_name = data.test_name;
    fprintf("Running %s...          ", test_name);
    
    % Sort the distances and corresponding indices for comparison
    [my_D_sorted, my_sort_order] = sort(my_D, 2);  % Sort each row
    my_IDX_sorted = my_IDX(sub2ind(size(my_IDX), ...
                                   repmat((1:size(my_IDX, 1))', 1, size(my_IDX, 2)), ...
                                   my_sort_order));
    
    % Compare sorted matrices
    tolerance = 1e-10;
    if all(abs(my_D_sorted(:) - test_D(:)) < tolerance) && isequal(my_IDX_sorted, test_IDX)
        result = 0; % Success
    else
        result = 1; % Failure
    end
end