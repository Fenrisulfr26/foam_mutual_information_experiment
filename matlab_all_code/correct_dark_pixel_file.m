function outFile = correct_dark_pixel_file(inFile)
%CORRECT_DARK_PIXEL_FILE Correct one MAT file containing variable "hist".
%
% Usage:
%   outFile = correct_dark_pixel_file(inFile)
%
% The output is saved beside inFile with "_dark_corrected" appended to the
% original file name. All original variables are preserved; the variable
% "hist" is replaced by the corrected histogram.

    if nargin < 1 || isempty(inFile)
        error('Input MAT file path is required.');
    end

    if ~(ischar(inFile) || isstring(inFile))
        error('inFile must be a character vector or string scalar.');
    end

    inFile = char(inFile);
    if ~isfile(inFile)
        error('File not found: %s', inFile);
    end

    vars = whos('-file', inFile);
    varNames = {vars.name};
    if ~ismember('hist', varNames)
        error('File does not contain a variable named hist: %s', inFile);
    end

    data = load(inFile);
    [data.hist, correctionInfo] = correct_hot_dark_pixels(data.hist);
    data.darkCorrectionInfo = correctionInfo;
    data.darkCorrectionInfo.sourceFile = inFile;
    data.darkCorrectionInfo.correctedOn = datetime('now');

    [folder, baseName, ext] = fileparts(inFile);
    outFile = fullfile(folder, [baseName, '_dark_corrected', ext]);
    save(outFile, '-struct', 'data', '-v7');
end
