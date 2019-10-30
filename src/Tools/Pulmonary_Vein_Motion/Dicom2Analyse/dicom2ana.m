function error = dicom2ana (input_pathname, output_pathname , gui_data)
%
% GUI version of Siemens Dicom to Analyze
%
% syntax:
% [error = ] dicom2ana[( input_pathname [, output_pathname ]) ]
%
% where:
% input_pathname is an optional parameter and is the directory where some 
% dicom data exists, either a DICOMDIR file, or some .IMA files
% input_pathname must be included is output_pathname is included
%
% output_pathname is an optional parameter and is the directory where 
% the analyze files will be deposited.  input_pathname must be included 
% is output_pathname is included
%
% examples:
% dicom2ana
% The user will be asked for the input and output directories
%
% dicom2ana ('C:\Data\Distortion_dir\')
% The user will be asked for the output directory
%
% dicom2ana ('C:\Data\Distortion_dir\' , 'C:\Data\Distortion_dir_output\');
% The user will not be asked for anything
%
% Known bugs:
% Directory selection is crude, so directories can only be selected by selecting a file
% in that directory when using the GUI.  This is bad because normally before you start
% there are no files in the output directory! 
%
% MDR 15/04/2002
% version 1.07
% Added some functionality so the code doesn't crash out when dicominfo cannot read the data
%
% version 1.08
% Added Tiff output capability: Note the TIFF images are scaled so that 1 pixel is 1mm
%
% version 1.09
% Matlab data output to a file containing all the patient data and slicing information
% compatible with data read functionality.
%
% version 1.10
% Catch the error associated with files not being present.  I think that this is introduced when acquisition 
% is started but aborted.  The data comes out alright anyway!
%
% version 1.11
% Additional traps to catch Argus like data in the reshape command (basically new stuff that I don't understand)
% and when Study doesn't exist for a patient (don't understand how this can happen) but it did!
%
% version 1.12
% Additional traps to handle patient names with ':' in them, also filenames and pathnames with ':'
%
% version 1.13
% Updated to handle 21A Mosaic files, and 15A Mosaic better (I think).  
% Also can handle single image mosaic
% Information on whether the file is Mosaic or not now comes from the looking in (Private_0029_1020) for '<ParamFunctor."MosaicUnwrapper">'
% Or under 15A looking for Mosaic in the imaging method
%
% Information regarding the number of images in the Moasic file is determined from looking at the blank images
% in the mosaic file (I agree its not very efficient, but it is gauranteed to work)
%
% Only images (from mosaic) with data in them are output
%
% Store some extra positional information in the matlab structure so that the image positions
% can be handled more easily
%
% version 1.14
% Updated to output the filename in the Patient.mat data relative to the location of the Patient.mat file
% (requested by Clare Jackson, John Van Aarde etc.)
% The Patient.mat file is also now named after the patient name/identifier
%
% version 1.15
% Updated with Clare Jacksons fix for UNIX systems (thanks Clare).  Primarily relaing to delimiters.
%
% version 1.16
% Updated with Stuart Clare fix for spaces in the patient name (problem under UNIX).
%
% version 1.17
% Bug for wrong delimiter corrected
% Bug for non axial orientations in mosaic data corrected (Rob)
% Bug for UNIX delimiter problems corrected (thanks to Stuart)
%
% version 1.18
% Pesky delimiters again.  Replace ':' with '-', also replace ';' with '-'
%
% version 1.19
% Cardiac functionality changed to allow TPM
% Basically any datafile that has the letters "TPM" in the title export to a format that can be read by the 
% Freiburg TPM code (i.e. use the dcm suffix on the data)
% Freiburg TPM code.  Put it all into a single TPM directory(i.e. use the dcm suffix on the data)
%
% version 1.19a
% Middle name no longer required parameter
%
% version 1.20
% Fixed a bug that resulted in only one image being output from a single Mosaic acquisition
% Changed the code so that you have to flag if you want the dicom data to be copied
%
% version 1.22
% Included functionality to interface with the gui
%
% version 1.23
% Loads of hacks to make things work properly with flags for different outputs
%
% vbersion 1.24
% Change the lategd flag (from d2a_gui) so that we can also have a "matching" text
% so look for this text, and create a directory with this name and copy the appropriate files into that format 
% using the TPM format
%
% version 1.25
% Bug fixed to allow lategd files to work properly with filenames that
% include nasty characters
%
% version 1.26
% Problem with Whole_Stucture and PatientID, fixed
%
% version 1.27
% Added the capability to output glmax/glmin into the analyse output
%
% version 1.28
% glmax and glmin should be as before, calmin and calmax are the ones that
% needed setting
%
% version 1.29
% finger trouble with the above, now fixed.
%
% version 1.30
% Increased robustness for CD's with data acquired on different days and
% maybe 12B version of the software
%
% version 1.32
% SliceLocation sometimes doesn't exist, so put in a test to check whether
% this is in the file
%
global analyze_data tiff_data Patient dicom_copy gui_flag gui_handle tpm_file lategd_file lategd_filename




if (ispc)
    %This means that the machine is a PC and presumably the delimiter is a '\' for filenames
    delimiter = '\';
    wrong_delimiter = '/';
else
    if (isunix == 0)
        % this machine is neither Windows or UNIX, RUN AND HIDE
        error = 'Machine is neither Windows or UNIX';
        return
    else
        delimiter = '/';
        wrong_delimiter = '\';
    end
end
delimiter;
nargin;

if (nargin < 1)
    [input_filename, input_pathname] = uigetfile ('*','Pick a directory for the DICOM input (by selecting a file in that directory)');
end
if ( isequal(exist(input_pathname,'dir'),0) )
    disp2('Path does not exist');
    error = 'input pathname does not exist';
    return
end
if (isempty(findstr(input_pathname(length(input_pathname)),delimiter)))
    input_pathname = [input_pathname delimiter];
end


if (nargin < 2)
    [output_filename, output_pathname] = uigetfile ('*','Pick an directory for Analyze output (by selecting a file in that directory)');
end

version = '1.32';

if (nargin == 3)
    %GUI is being used so take the flags from that and put them into this code
    analyze_data     = gui_data.analyze_data;
    tiff_data        = gui_data.tiff_image;
    tpm_file         = gui_data.tpm_file;
    dicom_copy       = gui_data.copy_dicom;
    lategd_file      = gui_data.lategd_file;
    lategd_filename  = gui_data.lategd_filename;
    matlab_file      = gui_data.matlab_file;
    gui_flag = 1;
    gui_handle = gui_data.text_handle;
    
else%Matlab data is automatically saved which described the file organisation of the DICOM data
    analyze_data = 1;   %Put this to 1 if you want to save Analyze data
    tiff_data    = 0;   %Put this to 1 if you want to save tiff images: Note tiff images are set to 1mm resolution
    tpm_file     = 1;
    dicom_copy   = 0;   %Flag this as one if you want to duplicate the dicom files
    lategd_file  = 0;
    lategd_filename = 'nothing';
    matlab_file  = 0;
    gui_flag = 0;
    
end

disp2 ( ['version: ' version ] );
error = 'no errors (were logged)';

colons = strfind(output_pathname, ':')
if (length(colons)  <= 1)
    % There is always one colon in the pathname (i.e. C:/...
    % pathname not changed
else
    output_pathname(colons(2:length(colons))) = '-';
    disp2(['Colons found in output pathname, therefore these have been swapped for - ']);
end

if ( isequal(exist(output_pathname,'dir'),0) )
    disp2('Output Path does not exist, creating');
    error = 'output pathname did not exist but was created';
    %
    % Strip off tailing delimiter
    if (isempty(findstr(output_pathname(length(output_pathname)),delimiter)) == 0)
        output_pathname = output_pathname(1:length(output_pathname)-1);
    end
    %Now split the directory into the pre-exising bit and the new bit
    tmp = findstr(output_pathname, delimiter)
    preexisting = output_pathname(1: tmp(length(tmp)));
    newbit = output_pathname(tmp(length(tmp))+1: length(output_pathname));
    if (isequal(exist(preexisting,'dir'),0) )
        disp2('The directory above the directory that the output file is to be created in doesnt exist');
        error = 'The directory above the directory that the output file is to be created in doesnt exist: Failed';
        return
    end
    x = mkdir( preexisting , newbit);
end
if (isempty(findstr(output_pathname(length(output_pathname)),delimiter)) )
    output_pathname = [output_pathname delimiter];
end

tic
% Read the data into matlab
Patient = MDR(input_pathname , delimiter, wrong_delimiter);



% Loop through the different Patients/Studies and Series dumping the data
% Label the directories with 
% Patient_1
% Study_1
% Series_1
% Put additional information on the data in a info.txt file in each directory
%
for patient_counter = 1 : length(Patient)
    if (isempty(Patient{patient_counter}) == 0)
        % Check the Patient ID for dodgy characters and replace them
        Patient{patient_counter}.PatientID = legitimise_filename(Patient{patient_counter}.PatientID);
        
        x = mkdir( output_pathname, [ 'Patient_' Patient{patient_counter}.PatientID ]);
        if (x == 0)
            disp2(['Directory ' , output_pathname, [ 'Patient_' Patient{patient_counter}.PatientID ] , '  Failed to be created']);
        end
        patient_dir = [output_pathname  'Patient_' Patient{patient_counter}.PatientID delimiter ];
        % Make the output name that we use here relative to the location of the Patient.mat file
        relative_name = patient_dir(length(output_pathname)+1:length(patient_dir));
        Patient{patient_counter}.stored_dir = relative_name;
        
        
        
        if (isempty(strmatch('Study' , fieldnames(Patient{patient_counter}))))
            % This means that there are no Studies in this Patient, I don't understand how this can
            % happen, but it crashed the code before, so I had better check for it
            disp2(['No Study in Patient ' num2str(patient_counter) ' not saving!']);
        else
            for study_counter = 1 : length(Patient{patient_counter}.Study)
                if (isempty(Patient{patient_counter}.Study{study_counter}) == 0)
                    x = mkdir( patient_dir, [ 'Study_' num2str(study_counter)]);
                    if (x == 0)
                        disp2(['Directory ' , patient_dir, [ 'Study_' num2str(study_counter)], '  Failed to be created']);
                    end
                    study_dir = [patient_dir 'Study_' num2str(study_counter) delimiter ];
                    
                    % Make the output name that we use here relative to the location of the Patient.mat file
                    relative_name = study_dir(length(output_pathname)+1:length(study_dir));
                    Patient{patient_counter}.Study{study_counter}.stored_dir = relative_name;
                    
                    for series_counter = 1 : length(Patient{patient_counter}.Study{study_counter}.Series)
                        if (isempty(Patient{patient_counter}.Study{study_counter}.Series{series_counter}) == 0)
                            x = mkdir( study_dir, [ 'Images_' num2str(series_counter)]);
                            if (x == 0)
                                disp2(['Directory ' , patient_dir, [ 'Study_' num2str(study_counter)], '  Failed to be created']);
                            end
                            images_dir = [study_dir 'Images_' num2str(series_counter) ];
                            % Make the output name that we use here relative to the location of the Patient.mat file
                            relative_name = images_dir(length(output_pathname)+1:length(images_dir));
                            Patient{patient_counter}.Study{study_counter}.Series{series_counter}.stored_dir = relative_name;
                            series_dir =  study_dir ;
                            try
                                output_filename = [ series_dir delimiter 'images_' num2str(series_counter) '_' Patient{patient_counter}.Study{study_counter}.Series{series_counter}.Whole_Structure.ProtocolName];
                                disp2(['output_filename = ' [ series_dir delimiter 'images_' num2str(series_counter) '_' Patient{patient_counter}.Study{study_counter}.Series{series_counter}.Whole_Structure.ProtocolName] ]);
                                output_filename = [ series_dir delimiter 'images_' num2str(series_counter) '_' Patient{patient_counter}.Study{study_counter}.Series{series_counter}.Whole_Structure.ProtocolName];
                                disp2(['output_filename = ' [ series_dir delimiter 'images_' num2str(series_counter) '_' Patient{patient_counter}.Study{study_counter}.Series{series_counter}.Whole_Structure.ProtocolName] ]);
                                
                            catch
                                %I will assume that the failure was due to the
                                %lack of a ProtocolName
                                
                                output_filename = [ series_dir delimiter 'images_' num2str(series_counter) '_NoProtocolName'];
                                disp2(['output_filename = ' [ series_dir delimiter 'images_' num2str(series_counter) '_NoProtocolName'] ]);
                            end
                            %Now dump the data in this series into this directory
                            %M2A
                            nothing = M2A (delimiter , output_filename , output_pathname , series_dir, images_dir , patient_counter , study_counter , series_counter );
                            
                        end
                    end
                end
            end
        end
    end
end

% MATLAB output filename


if (length(Patient) == 1)
    %If there is only one patient, then use that name as the matlab filename.
    %Only use the forst 40 characters of the name (some MATLAB feature)
    tmp = ['Patient_' Patient{1}.PatientID];
    output_matlab_name = [output_pathname tmp(1 : min(length(tmp),40))];
else
    %If there are multiple then use the above and _plus_N where "N" is the number of other patients
    tmp = ['Patient_' Patient{1}.PatientID '_plus_' num2str(length(Patient)-1)];
    %Only use the forst 40 characters of the name (some MATLAB feature)
    output_matlab_name = [output_pathname tmp(1 : min(length(tmp),40)) ];
end

if (matlab_file == 1)
    save ( [ output_matlab_name '.mat'] , 'Patient')
end

disp2(';-)');
toc
return

function Patient = MDR (directory, delimiter, wrong_delimiter)
% MDR
%
% Matlab Dicom Reader
%
% Written by MDR for reading Siemens dicom data into Matlab
%
% Started on 25thMarch 2002
%
global analyze_data tiff_data
tic

if (nargin < 1)
    disp2('MDR called with no input directory');
    return;
end

filenames_list = dir (directory);


fprintf (['Looking at Directory: ' directory '  sniffing for a DICOMDIR file\n']);
dicomdir_found = 0;
IMAfiles_found = 0;
for counter = 1 : size(filenames_list,1)
    if ( strcmp(filenames_list(counter).name , 'dicomdir') == 1  | strcmp(filenames_list(counter).name , 'DICOMDIR') == 1) 
        fprintf('DICOMdir found, and being used\n');
        dicomdir_found = 1;
    end
    occurences_of_IMA = strfind(filenames_list(counter).name , '.IMA');
    if (length(occurences_of_IMA) ~= 0)
        if(occurences_of_IMA(length(occurences_of_IMA)) == length(filenames_list(counter).name)-3  )
            % the filename ends in IMA
            if (IMAfiles_found ~= 1)
                fprintf('IMAfiles found, these will be used if there is no DICOMdir\n');
                IMAfiles_found = 1;
            end
        end
    end
    
end


if (dicomdir_found == 1)
    fn = strcat(directory , 'dicomdir');
    
    % First question.  What are the datafiles within this directory?
    fprintf('Reading DICOM info');
    info = dicominfo (fn);
    
    field_names = fieldnames(info.DirectoryRecordSequence);
    
    Patient_counter = 1;
    
    Study_counter = 1;
    
    Series_counter = 1;
    
    
    Max_images = length(field_names);
    Image = zeros(Max_images,1);
    Image_counter = 1;
    Image_filename_number = zeros(Max_images,1);
    
    for counter = 1 : length(field_names)
        
        whatisit = eval( strcat ('info.DirectoryRecordSequence.Item_',int2str(counter),'.DirectoryRecordType') );
        %whatisit will either contain:
        % IMAGE
        % SERIES
        % PATIENT
        % or
        % STUDY
        flag = 0;
        if ( strcmp(whatisit , 'PATIENT') == 1 ) 
            flag = 1;
            % Put this information into the arrays for processing
            Patient_data_item(Patient_counter)   = counter;
            % I am assuming that Study occur in order, as there seems to be no reference back to the patientID from the Study
            % therefore this temporary variable should help me out to link Patients with Studies
            tmp_PatientID = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.PatientID'));
            Patient_data_PatientID{Patient_counter} = tmp_PatientID;
            
            Patient_data_Item(Patient_counter) = counter;
            Patient_counter = Patient_counter + 1;
            fprintf(strcat ('\n\nPatient--' , eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.PatientID')) ) );
        end
        if ( strcmp(whatisit , 'STUDY') == 1 ) 
            flag = 1;
            % Put this information into the arrays for processing
            Study_data_Item(Study_counter)    = counter;
            %As above there appears to be no clear link between the Study and the Series, therefore I am going to assume that they 
            %appear in this file in a causal fashion (I think that this might be wrong, but I don't yet understand the conventions)
            tmp_StudyNumber = str2num(eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.StudyID')));
            Study_data_Study(Study_counter) = tmp_StudyNumber;
            Study_data_PatientID{Study_counter} = tmp_PatientID;
            
            fprintf (strcat('\n  study--' , eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.StudyDescription')) , '--' ,num2str(eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.StudyID'))) ) );
            
            Study_counter = Study_counter + 1;
        end
        if ( strcmp(whatisit , 'SERIES') == 1 )
            flag = 1;
            %Put this information into the arrays for processing
            Series_data_Item(Series_counter)    = counter;
            Series_data_Series(Series_counter,:) = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.SeriesNumber'));
            Series_data_Study(Series_counter,:) = tmp_StudyNumber;
            Series_data_PatientID{Series_counter} = tmp_PatientID;
            
            Series(Series_counter) = counter;
            
            fprintf (strcat('\n  series--' , eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.SeriesDescription')) , '--' ,num2str(eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.SeriesNumber'))) ) );
            
            Series_counter = Series_counter + 1;
        end
        
        % if ( strcmp(whatisit , 'PRIVATE') == 1 ) 
        %    flag = 1;
        %      fprintf (strcat('\n  Spectroscopy--' , eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(counter), '.SeriesDescription')) ) );
        % end
        
        if ( strcmp(whatisit , 'IMAGE') == 1 ) 
            flag = 1;
            
            Image(Image_counter) = counter;
            
            %file_directory = strcat(directory , eval( strcat ('info.DirectoryRecordSequence.Item_',int2str(counter),'.ReferencedFileID') ) );
            dicom_filename = eval(strcat('info.DirectoryRecordSequence.Item_',int2str(counter),'.ReferencedFileID') ); 
            pos_delimiter = findstr(dicom_filename , wrong_delimiter);  %This swaps the PC to the UNIX delimiter, or vice versa 
            dicom_filename(pos_delimiter) = delimiter; 
            
            file_directory = strcat(directory , dicom_filename); 
            
            %
            % So, it appears that the images appear in an arbitrary sequence within the DICOM file!  Shit or what?
            %
            % So, we need to look into each dicom image file to determine which series the file is in and which study and instance
            % I don't understand the significance of these numbers (as yet) but hopefully it will all become clear!
            file_directory;
            try
                info_tmp = dicominfo(file_directory);
                
                
                %new_image = dicomread(Image_filename_tmp);
                %imagesc (new_image);
                
                %pause(0.1);
                %fprintf (strcat('\n  image--' , info_tmp.StudyID , '--' , num2str(info_tmp.SeriesNumber) , '--' , num2str(info_tmp.AcquisitionNumber) , '--' , num2str(info_tmp.InstanceNumber) ) );
                
                % Put this information into the arrays for processing
                
                Image_data_Item(Image_counter)     = counter;
                Image_data_Study(Image_counter)    = str2num(info_tmp.StudyID);
                Image_data_Series(Image_counter)   = info_tmp.SeriesNumber;
                Image_data_Image(Image_counter)    = info_tmp.AcquisitionNumber;
                Image_data_Image2(Image_counter)    = info_tmp.InstanceNumber;
                Image_data_Filename{Image_counter} = file_directory;
                Image_data_PatientID{Image_counter}  = info_tmp.PatientID;
                
                if (strcmp(info_tmp.PatientID , tmp_PatientID ) ==  0)
                    % PatientID is non-causal with respect to images, this might mean that this rule is also bogus for the series info
                    fprintf('Non-Causal PatientID in image data, worry a little (but not too much)\n');
                end
                
                %Create the structure that describes the format of all this data.
                %this should replicate the tree structure that is visible from the viewer (I believe that there is adequate information
                % in the above indices!
                
                Image_counter = Image_counter + 1;
            catch
                disp2(['MATLAB DICOMREAD error for ' file_directory ' skipping']);
            end
        end
        if (flag == 0)
            fprintf(strcat('New Item, not handled: ' , whatisit));
        end
        
    end
    
    %
    % Now I have the whole structure of the file and can do anything that I want with it.
    %
    % Notably I can find out the information on an individual file using
    % Image_filename_tmp = strcat(directory , int2str(Image_filename_number(XXX)) );
    % Where XXX is the number of the image and is between 1 >= XXX > Image_counter
    % info = dicominfo(Image_filename_tmp)
    % info gives access to all sorts of variables, names, orientations etc.
    %
    % For example the bottom corner of the image is given by
    % info.ImagePositionPatient
    % and the orientation of the image by
    % info.ImageOrientationPatient
    %
    % NOTE: it may be necessary to update your MATLAB Dicom installation (there are a few patches) from the www.matlab.com site
    %       just search on MATLAB
    %
    % Next you can read an image file with
    % new_image = dicomread(info);
    % imagesc , new_image
    
    % see for example
    
    %XXX = 4;
    %Image_filename_tmp = strcat(file_directory , int2str(Image_filename_number(XXX)) );
    %info_tmp = dicominfo(Image_filename_tmp);
    %new_image = dicomread(Image_filename_tmp);
    %imagesc (new_image);
    
    
    
    %
    % Now that we have everything, we can put it into the structure
    %    Patients(pc).Study( str2num(info_tmp.StudyID) ).Series( info_tmp.SeriesNumber ).Images( info_tmp.AcquisitionNumber ).filename = 
    
    disp2(' ');
    disp2('Converting into MATLAB Structured form');
    
    %
    % First go through the list of patients, get their reference numbers and put this information into the structure
    Number_of_Patients = Patient_counter-1;
    for Patient_counter = 1 : Number_of_Patients
        
        Patient{Patient_counter}.PatientID =  Patient_data_PatientID{Patient_counter};
        Patient{Patient_counter}.Item_No =  Patient_data_Item(Patient_counter);
        
        %We could tag the whole structure in here with
        Patient{Patient_counter}.Whole_Structure = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(Patient_data_item(Patient_counter))));
    end
    % Now add the study information
    
    Number_of_Studies = Study_counter -1;
    for Study_counter = 1 : Number_of_Studies
        
        %First find the patient number for this study
        for Patient_counter = 1 : Number_of_Patients
            if (strcmp ( Patient{Patient_counter}.PatientID , Study_data_PatientID{Study_counter}) == 1)
                Patient_Number = Patient_counter;
            end
        end
        Study_Number = Study_data_Study(Study_counter);
        
        Patient{Patient_Number}.Study{Study_Number}.Whole_Structure = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(Study_data_Item(Study_counter))));
        Patient{Patient_Number}.Study{Study_Number}.Item_No = Study_data_Item(Study_counter);
        
    end
    
    % Now the series information
    
    Number_of_Series = Series_counter -1;
    for Series_counter = 1 : Number_of_Series 
        
        %First find the patient number for this Series
        for Patient_counter = 1 : Number_of_Patients
            if (strcmp ( Patient{Patient_counter}.PatientID, Series_data_PatientID{Series_counter}) == 1)
                Patient_Number = Patient_counter;
            end
        end
        Study_Number  = Series_data_Study(Series_counter);
        Series_Number = Series_data_Series(Series_counter);
        
        Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Whole_Structure = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(Series_data_Item(Series_counter))));
        Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Item_No         = Series_data_Item(Series_counter);
        Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.SeriesDescription        = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(Series_data_Item(Series_counter)), '.SeriesDescription'));
        
    end
    
    %Finally the image details
    
    Number_of_Images = Image_counter -1;
    for Image_counter = 1 : Number_of_Images 
        
        %First find the patient number for this Images
        for Patient_counter = 1 : Number_of_Patients
            if (strcmp ( Patient{Patient_counter}.PatientID , Image_data_PatientID{Image_counter}) == 1)
                Patient_Number = Patient_counter;
            end
        end
        Study_Number  = Image_data_Study(Image_counter);
        Series_Number = Image_data_Series(Image_counter);
        Image_Number = Image_data_Image2(Image_counter);
        %directory
        %Image_data_Filename
        file_directory = [ char(Image_data_Filename(Image_counter))];
        info_tmp = dicominfo(file_directory);
        
        %
        argus_flag = 0;
        if (isempty(strmatch('SecondaryCaptureDeviceID' , fieldnames(info_tmp))) == 0)
            % No SecondaryCaptureDeviceID (that means that it can't be Argus) [to me]
            if (strcmp(info_tmp.SecondaryCaptureDeviceID, 'Argus'))
                % This data is an Argus output file,  don't even think about reading it in
                disp2(['Argus data file at ' file_directory ' skipping']);
                argus_flag = 1;
            end
        end
        
        if (argus_flag == 0)
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.Rows = info_tmp.Rows;
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.Columns = info_tmp.Columns;
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.PixelSpacing = info_tmp.PixelSpacing;
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.ImagePositionPatient = info_tmp.ImagePositionPatient;
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Whole_Structure = info_tmp;
            if(~isempty(strmatch('SliceLocation' , fieldnames(Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure))))
                Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.SliceLocation = info_tmp.SliceLocation;
            end
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.SliceThickness = info_tmp.SliceThickness;
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.ImageOrientationPatient = info_tmp.ImageOrientationPatient;
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.PatientPosition = info_tmp.PatientPosition;
            %
            % The orientation should contain 6 real numbers in an ascii string (cosines of the 
            % angles of the x and y axes of the image between the x, y and z axes of the 
            % patient, and the position should contain 3 real numbers (3D-coordinates).
            % Patient_x_coord = ImageOrientationPatient(1) * x_image + ImageOrientationPatient(4) * y_image
            % Patient_y_coord = ImageOrientationPatient(2) * x_image + ImageOrientationPatient(5) * y_image
            % Patient_z_coord = ImageOrientationPatient(3) * x_image + ImageOrientationPatient(6) * y_image
            
            
            %Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(Image_data_Item(Image_counter))));
            
            %Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(Image_data_Item(Image_counter))));
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Item_No          = Image_data_Item(Image_counter);
            Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Filename         = Image_data_Filename(Image_counter);
        end
    end
    
    
elseif (IMAfiles_found == 1)
    
    %
    % Plan B
    %
    % Loop through the IMA files putting the information into the 
    % Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number} structure (as before)
    patients = 0;
    for counter = 1 : length(filenames_list)
        
        occurences_of_IMA = strfind(filenames_list(counter).name , '.IMA');
        if (length(occurences_of_IMA) ~= 0)
            if(occurences_of_IMA(length(occurences_of_IMA)) == length(filenames_list(counter).name)-3  )
                % the filename ends in IMA
                fprintf([filenames_list(counter).name '\n']);
                %Processing this file
                
                file_directory = [directory filenames_list(counter).name];
                info_tmp = dicominfo(file_directory);
                
                
                %Is this spectropscopy DATA ?
                
                % The spectroscopy label appears in Private_0029_1008 (in most of my data I think!).
                % If that field exists check to see that the data is image data,
                % if the field doesn't exist then lets assume that the data is image data and crack on
                
                spectroscopy_flag = 1;
                exist_flag = 1;
                if (isempty(strmatch('Private_0029_1008' , fieldnames(info_tmp))))
                    % Private_0029_1008 does not exist
                    % This probably mean that the data has been acquired from pre-exisintg data (i.e. by processing)
                    exist_flag = 0;
                else
                    % Private_0029_1008 does exist
                    % therefore see if is spectroscopy or imaging
                    if (strcmp(info_tmp.Private_0029_1008  , 'IMAGE NUM 4') ~= 0)
                        % Image data
                        spectroscopy_flag = 0;
                    end
                end
                
                if (exist_flag == 0)
                    % This data is data that is from a strange source (i.e. probably processed somehow)
                    fprintf(['Data from wierd source at ' filenames_list(counter).name ' skipping\n']);
                elseif (spectroscopy_flag == 1)
                    % This data is spectroscopy
                    fprintf(['Spectroscopy data at ' filenames_list(counter).name ' skipping\n']);
                else
                    
                    %if (counter == 1)
                    %   Patients{1}.PatientID = info_tmp.PatientID;
                    %endNONIMAGE NUM 4
                    %Find which patient we are looking at
                    Patient_Number = 0;
                    for patient_counter = 1 : patients
                        if (info_tmp.PatientID == Patient{patient_counter}.PatientID)
                            Patient_Number = patient_counter;
                        end
                    end
                    %Patient not found?
                    if (Patient_Number == 0)
                        % Add a new patient
                        patients = patients + 1;
                        Patient{patients}.PatientID = info_tmp.PatientID;
                        Patient_Number = patients;
                    end
                    
                    
                    Study_Number     = str2num(info_tmp.StudyID);
                    Series_Number    = info_tmp.SeriesNumber;
                    Image_data_Image     = info_tmp.AcquisitionNumber;
                    Image_data_Image2   = info_tmp.InstanceNumber;
                    
                    
                    Image_Number = Image_data_Image2;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.Rows = info_tmp.Rows;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.Columns = info_tmp.Columns;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.PixelSpacing = info_tmp.PixelSpacing;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.ImagePositionPatient = info_tmp.ImagePositionPatient;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Whole_Structure = info_tmp;       
                    %Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure = eval(strcat('info.DirectoryRecordSequence.Item_' , int2str(Image_data_Item(Image_counter))));
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.ImagePositionPatient = info_tmp.ImagePositionPatient;
                    if(~isempty(strmatch('SliceLocation' , fieldnames(Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure))))
                        Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.SliceLocation = info_tmp.SliceLocation;
                    end
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.SliceThickness = info_tmp.SliceThickness;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.ImageOrientationPatient = info_tmp.ImageOrientationPatient;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Whole_Structure.PatientPosition = info_tmp.PatientPosition;
                    
                    
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Item_No         = -1;
                    Patient{Patient_Number}.Study{Study_Number}.Series{Series_Number}.Image{Image_Number}.Filename        = file_directory;
                end
            end
        end
    end
    
else
    disp2('No DICOMdir or IMA files found in the directory, try a different location\n');    
end

toc

return


function nothing = M2A ( delimiter , output_filename , output_pathname , output_dirname , images_dir , patient_no , study , series , image )
%
% Write a set of images to an analyze file
%
% function VO = M2A ( delimiter , output_filename , output_dirname , patient_no , study , series , image )
%
% This takes a range of data (i.e.   M2A(dicom_str, 1 , 1, 1:2 , 1:128) ) and
% bungs it into an Analyze file format.  If the data are not of a consistent size then
% this probably will work but the output will be ugly!
%
% dicom_str = dicom structure (from MDR.m) this is a cell structure that contains all the information on the dicom data
% output_filename = filename for the output data (note that there will be a 'output_filename'.hdr & an 'output_filename.img'
% output_pathname = pathname of where the data is going to be output to
% patient_no   = integer (or array of integers) describing which data to dump
% study     = as above
% image     = as above (optional argument)
%
%    flip_flag = 0; % Don't flip            (dimensions remain the same)
%    flip_flag = 1; % flip diagonally       (dimensions flip)
%    flip_flag = 2; % rotate by 90 degrees  (dimensions flip)
%    flip_flag = 3; % rotate by 180 degrees (dimensions remain the same)
%    flip_flag = 4; % rotate by 270 degrees (dimensions flip)
%    flip_flag = 5; % flip up down          (dimensions remain the same)
%    flip_flag = 6; % flip left right       (dimensions remain the same)
%
% Note: if the image argument is not included, all the images in that series will be used
%
% This code can also handle MOSAIC data
%
% MDR 1/04/2002
%
nothing = 0;
flip_flag = 4;
global analyze_data tiff_data Patient dicom_copy tpm_file lategd_file lategd_filename


cal_min =   10000000000;
cal_max =  -10000000000;

if (nargin < 8) 
    fprintf('M2A: called with too few arguments\n');
    return;
end
if (nargin == 8)
    % M2A called without 'images' argument, assume that all the images should be included
    
    %Sometimes there are no images!
    %when this is the case
    
    if (isempty(strmatch('Image' , fieldnames(Patient{patient_no}.Study{study}.Series{series}))) == 1)
        disp2(['No images in Series ' num2str(series) ]);
        error = ['No images in Series ' num2str(series) ];
        return;
    else
        % This is the way that it used to work with 15A
        %image = 1:length(Patient{patient_no}.Study{study}.Series{series}.Image);
        
        %New method, compatible with 21A (and 15A)
        %Count the number of images that we have
        number_of_actual_images = 0;
        for image_counter = 1 : length(Patient{patient_no}.Study{study}.Series{series}.Image)
            if (isstruct(Patient{patient_no}.Study{study}.Series{series}.Image{image_counter}))
                number_of_actual_images = number_of_actual_images + 1;
            end
        end
        image = zeros(1,number_of_actual_images );
        number_of_actual_images = 0;
        for image_counter = 1 : length(Patient{patient_no}.Study{study}.Series{series}.Image)
            if (isstruct(Patient{patient_no}.Study{study}.Series{series}.Image{image_counter}))
                number_of_actual_images = number_of_actual_images + 1;
                image(1,number_of_actual_images) = image_counter;
            end
        end
        
        
    end
end


if (analyze_data == 1)
    % Open the image file
    fname = strcat(output_filename , '.img');
    fp    = fopen(fname,'r+');
    if fp == -1,
        fp = fopen(fname,'w');
        if fp == -1,
            VO = [];
            disp2('Can''t open image file');
            error = 'Can''t open image file';
            return;
        end
    end
end

Mosaic_data = 0;
% Note this main section will only work on NON-MOSAIC datasets
% The following doesn't guarantee that it is Mosaic! 
%fieldnames(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)})
if (isempty(strmatch('Image' , fieldnames(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}))))
    % No images in this series, so we don't want the following if to occur
else
    % This code was necessary for the old MOSAIC form which had loads of gaps in the image list
    if ( isstruct(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Image{1}) == 0) 
        % The first image is empty so this may be a MOSAIC dataset this will be handled later
        % MOSIAC datasets are characterised by  (N*N)-1 empty images, followed by
        % a very full image.  
        % This is repeated for the whole dataset
        % The numbering of the files was as above for 15A, but in 21A they changed this to remove all the empty images!
        %fprintf('MOSAIC data\n');
        
        % Lets find out about the MOSAIC format
        flag = 0;
        for tmp_counter = 1 : length(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Image)
            if ( isstruct (Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Image{image(tmp_counter)}) == 1)
                % Find the first image that exists as some real data
                break
            end
        end
        if (tmp_counter == length(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Image) )
            % If this first image appears as the last image in the data then
            % This is some kind of composite dataset (like a t-map)
            fprintf('Composite dataset: aborting M2A\n');
            disp2(['Image type ' , Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.ImageType ]);
            return;
        end
        
    end     
    % So, the first actual data is at "tmp_counter"
    % The format of the MOSAIC data is such that:
    % if there are N images in the mosaic
    % the data will be at N-1, (2N)-1, (3N)-1, etc.
    %
    % The Mosaic file seems also always to be square (NOT TRUE)
    
    
    %Mos_factor = sqrt(tmp_counter+1);
    %fprintf(['Mosaic factor ' num2str(Mos_factor) 'x' num2str(Mos_factor) '\n']);
    %if (Mos_factor == floor(Mos_factor))
    
    if (isempty(strmatch('Whole_Structure',fieldnames(Patient{patient_no(1)}))))
        % Create a Whole_Structure for the patient
        Patient{patient_no(1)}.Whole_Structure.PatientID = Patient{patient_no(1)}.PatientID; %This does seem to be mindless duplication!
        Patient{patient_no(1)}.Whole_Structure.PatientsName.FamilyName = 'not_recorded';
        Patient{patient_no(1)}.Whole_Structure.PatientsName.GivenName = 'not_recorded';
    end
    
    MOSAIC_FLAG = 0;
    
    if (strcmp(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.ImageType , 'ORIGINAL\PRIMARY\M\MOSAIC'))
        MOSAIC_FLAG = 1;
    end
    
    if (strcmp(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.ImageType , 'ORIGINAL\PRIMARY\M\ND\MOSAIC')) 
        MOSAIC_FLAG = 1;
    end
    if (~isempty(strmatch('Private_0029_1020' , fieldnames(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure))))
        if (~isempty(findstr('<ParamFunctor."MosaicUnwrapper">', char(Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.Private_0029_1020)')))
            MOSAIC_FLAG = 1;
        end
    end
    
    if (MOSAIC_FLAG == 1)
        % This looks like a mosaic dataset 
        % The first case finds it under 15A
        % The second case finds it under 21A (bastards)
        %
        
        %The following code is required to handle non-axial scans
        if (Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.AcquisitionMatrix(1)  ~= 0)
            Mos_factor = Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.Columns   / Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.AcquisitionMatrix(1);
        else
            if (Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.AcquisitionMatrix(2)  ~= 0)
                Mos_factor = Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.Columns   / Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.AcquisitionMatrix(2);
            else
                Mos_factor = Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.Columns   / Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.AcquisitionMatrix(3);
            end
        end
        
        %
        %Mos_factor = Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.Rows / Patient{patient_no(1)}.Study{study(1)}.Series{series(1)}.Whole_Structure.AcquisitionMatrix(4);
        
        Mosaic_data = 1;
    end
    
    
end
data_output = 0; % A flag to indicate whether data has been output, and which tells whether to output a header
if (Mosaic_data == 1)
    
    flag = 0;
    counter = 1;
    for i = patient_no
        for j = study
            for k = series
                Patient{i}.Study{j}.Series{k}.Total_Images = 0;
                for l = image
                    %tmp = [i , j , k , l]
                    if ( isstruct (Patient{i}.Study{j}.Series{k}.Image{l}) == 1)
                        %Only do anything at all if there is data present!
                        fprintf('.');
                        if (flag == 2)
                            flag = 3;
                        end
                        if (flag == 1)
                            %Now I have the second slice I can determine the slice to slice distance (i.e. z_pixel dimension)
                            % actually this is included in the SpacingBetweenSlices element of the structure
                            %Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient
                            
                            % This slice vector functionality doesn't work for mosaic data
                            %slice_vector = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient - z_slice_posn;
                            flag = 2;    
                        end
                        if (flag == 0)
                            %Data dimensions
                            dim = zeros(8,1);
                            
                            dim(1) = 4; % number of dimensions
                            dim(2) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.Rows / Mos_factor;   
                            dim(3) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.Columns / Mos_factor;
                            dim(4) = 1; %This is a placeholder incase there is only one image (normally this will be overwritten)
                            %
                            % Look for sSliceArray.lSize in the Pricate information file
                            %  This proved unreliable too!
                            %field names
                            %private_text = char(Patient{1}.Study{1}.Series{2}.Whole_Structure.Private_0029_1020)';
                            %position_of_lSize_information = findstr(private_text , 'sSliceArray.lSize');
                            %residual = private_text(position_of_lSize_information:position_of_lSize_information+100);
                            %[token ,residual] = strtok(residual);
                            %%token will be 'sSliceArray.lSize'
                            %[token ,residual] = strtok(residual);
                            %%token will be '='
                            %[token ,residual] = strtok(residual);
                            %Actual_number_of_used_slices = str2num(token);
                            %
                            %dim(4) = Actual_number_of_used_slices;
                            %
                            %We will have to configure dim(4) later
                            
                            
                            %In this case there is one volume per image in the 
                            dim(5) = (length(patient_no) * length(study) * length(series) * length(image)); 
                            
                            %dim(4) = Mos_factor*Mos_factor; %Ironically this is correct, but only some of the boxes in the output file are filled!
                            %dim(5) = (length(patient_no) * length(study) * length(series) * length(image)+1)/(Mos_factor*Mos_factor); %This has been assumed but we probably need to be cleverer about dim(4) and dim(5)
                            dim(6) = 1; % only 4D therefore irrelevant
                            dim(7) = 1; % only 4D therefore irrelevant
                            dim(8) = 1; % only 4D therefore irrelevant
                            
                            pixdim = zeros(8,1);
                            pixdim(1) = 0;
                            pixdim(2) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.PixelSpacing(1); % these may be the wrong way around
                            pixdim(3) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.PixelSpacing(2); %these may be the wrong way around
                            %I can only get the z_pixel spacing from the interslice distance (how do I do this for Mosaic?)
                            %pixdim(4) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.SpacingBetweenSlices;
                            %pixdim(5) = 1; %This is arbitrary
                            pixdim(6) = 1; % only 4D therefore irrelevant
                            pixdim(7) = 1; % only 4D therefore irrelevant
                            pixdim(8) = 1; % only 4D therefore irrelevant
                            
                            %A = zeros(dim(2)*dim(3)*dim(4)*dim(5),1);
                            A = zeros(dim(2)*dim(3),1);
                            
                            %flip_flag = 0; % Don't flip            (dimensions remain the same)
                            %flip_flag = 1; % flip diagonally       (dimensions flip)
                            %flip_flag = 2; % rotate by 90 degrees  (dimensions flip)
                            %flip_flag = 3; % rotate by 180 degrees (dimensions remain the same)
                            %flip_flag = 4; % rotate by 270 degrees (dimensions flip)
                            %flip_flag = 5; % flip up down       (dimensions remain the same)
                            %flip_flag = 6; % flip left right    (dimensions remain the same)
                            if (flip_flag == 1 | flip_flag == 2 | flip_flag == 4)
                                %Swap the dimensions 
                                store_dim = dim;
                                temp = dim(2);
                                dim(2) = dim(3);
                                dim(3) = temp;
                                % And the voxel sizes
                                temp = pixdim(2);
                                pixdim(2) = pixdim(3);
                                pixdim(3) = temp;
                            end
                            
                            %Snag the slice orientation information for the aux file
                            pixel_origin = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient;       % 3 floats
                            %Next the vector describing how to get from one slice to the next
                            slice_vector = [0.0 , 0.0 , 0.0 ]; %this will be created when flag == 1 (when we get to the next slice)
                            %Next the in-plane vector for the data as it is in the Dicom file
                            read_vector = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImageOrientationPatient(1:3);% 3 floats for read direction
                            phase_vector  = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImageOrientationPatient(4:6);       % 3 floats for phase encode direction
                            %This describes the orientation of the patient within the magnet
                            Patient_Position = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.PatientPosition;
                            %This is probable extra information, but might be useful
                            %slice_location = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.SliceLocation;
                            %Finally the information concerning whether the images have been flipped in the output file
                            %flip_flag
                            
                            flag = 1    ;             
                        end
                        
                        full_filename = char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename);
                        tmp_image = dicomread( full_filename);
                        
                        %
                        % This is to copy the file to the same directory as the output data
                        % therefore, trim off the directory part of the filename
                        slashes = strfind( full_filename , delimiter);
                        
                        filename_bit = full_filename (slashes(length(slashes)) : length(full_filename));
                        
                        %
                        %
                        
                        if (tpm_file == 1 | lategd_file == 1)
                            if (isempty(strmatch('Whole_Structure',fieldnames(Patient{i}))))
                                % Create a WHole_Structure for the patient
                                Patient{i}.Whole_Structure.PatientID = Patient{i}.PatientID; %This does seem to be mindless duplication!
                                Patient{i}.Whole_Structure.PatientsName.FamilyName = 'not_recorded';
                                Patient{i}.Whole_Structure.PatientsName.GivenName = 'not_recorded';
                            end
                            if (isempty(strmatch('Whole_Structure',fieldnames(Patient{i}.Study{j}))))
                                Patient{i}.Study{j}.Whole_Structure.StudyDate = 'not_recorded';
                                
                            end        
                        end
                        
                        
                        
                        
                        if (dicom_copy == 1)
                            
                            copyfile( char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename) , [ images_dir delimiter filename_bit ] , 'writable');
                            
                        end
                        
                        Patient{i}.Whole_Structure.PatientID = legitimise_filename(Patient{i}.Whole_Structure.PatientID);
                        
                        if (isempty(strfind(Patient{i}.Study{j}.Series{k}.Whole_Structure.ProtocolName,'TPM')) & isempty(strfind(Patient{i}.Study{j}.Series{k}.Whole_Structure.ProtocolName,'tpm')))
                            % This is not a Tissue Phase Map file so carry on as normal           
                        else
                            if (tpm_file == 1)
                                %The Freiburg TPM code requires that '.dcm' is the suffix on all file names
                                % Also put the Frieburg TPM datafiles into a single TPM directory             
                                TPM_dir = [output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)  delimiter 'TPM' ];
                                if (isequal(exist(TPM_dir,'dir'),0) )
                                    mkdir([output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)  delimiter] , 'TPM');
                                end
                                %What should the output filename be?
                                % 
                                tmp_series_name = sprintf('00000%d',k);
                                tmp_image_name  = sprintf('00000%d',l);
                                try
                                TPM_filename = [Patient{i}.Whole_Structure.PatientsName.FamilyName '_' Patient{i}.Whole_Structure.PatientsName.GivenName '__' ...
                                        '_' Patient{i}.Study{j}.Whole_Structure.StudyDate ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                            catch
                                TPM_filename = ['not_recorded' ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                                
                            end
                                %MiddleName removed as it crashes when there is no middle name!
                                
                                
                                
                                copyfile( char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename) , [ TPM_dir delimiter TPM_filename] , 'writable');
                            end
                        end
                        
                        if (isempty(strfind(lower(Patient{i}.Study{j}.Series{k}.Whole_Structure.ProtocolName),lower(lategd_filename))))
                            % This is not a Tissue Phase Map file so carry on as normal           
                        else
                            if (lategd_file == 1)
                                %The Freiburg TPM code requires that '.dcm' is the suffix on all file names
                                % Also put the Frieburg TPM datafiles into a single TPM directory
                                LATEGD_dir = [output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)  delimiter lategd_filename ];
                                if (isequal(exist(LATEGD_dir,'dir'),0) )
                                    mkdir([output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)] , lategd_filename );
                                end
                                %What should the output filename be?
                                % 
                                tmp_series_name = sprintf('00000%d',k);
                                tmp_image_name  = sprintf('00000%d',l);
                                try
                                LATEGD_filename = [Patient{i}.Whole_Structure.PatientsName.FamilyName '_' Patient{i}.Whole_Structure.PatientsName.GivenName '__' ...
                                        '_' Patient{i}.Study{j}.Whole_Structure.StudyDate ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                            catch
                                LATEGD_filename = ['not_recorded' ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                            
                                %MiddleName removed as it crashes when there is no middle name!
                            end
                                
                                
                                copyfile( char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename) , [ LATEGD_dir delimiter LATEGD_filename] , 'writable');
                            end
                        end
                        
                        
                        % Make the output name that we use here relative to the location of the Patient.mat file
                        relative_name = images_dir(length(output_pathname)+1:length(images_dir));
                        Patient{i}.Study{j}.Series{k}.Image{l}.stored_dir =  [ relative_name delimiter filename_bit ];
                        
                        
                        
                        if (flag == 1) 
                            info = dicominfo(  char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename));
                            if (isempty(strmatch('SpacingBetweenSlices' , fieldnames(info))))
                                pixdim(4) = 0.0;
                            else
                                pixdim(4) = info.SpacingBetweenSlices;
                            end
                            time_of_first_slice = str2num(info.AcquisitionTime);
                        end
                        if (flag == 2)
                            info = dicominfo(  char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename));
                            pixdim(5) = str2num(info.AcquisitionTime) - time_of_first_slice;
                            
                            if (pixdim(5) < 0)
                                %Data acquired at mid-night or midday, may get buggered
                                pixdim(5) = pixdim(5) +  120000.0;
                            end
                            if (pixdim(5) > 4000.0)
                                %it is likely that this has gone over a "hour" boundary
                                % Time is stored as a character array of
                                pixdim(5) = pixdim(5) - 4000.0;
                            end
                            if (pixdim(5) > 40.0)
                                % It is likely that this has gone over a "minute" boundary
                                % Time is stored as a character array of
                                pixdim(5) = pixdim(5) - 40.0;
                            end
                        end
                        counter = 1;
                        image_output_counter = 0;
                        for x = 1 : Mos_factor
                            for y = 1 : Mos_factor
                                
                                
                                try
                                    sub_image = tmp_image((1+(store_dim(2)*(x-1))):(store_dim(2)*x),(1+(store_dim(3)*(y-1))):(store_dim(3)*y));
                                catch
                                    disp2(['I dont get it']);
                                end
                                %flip_flag = 0; % Don't flip            (dimensions remain the same)
                                %flip_flag = 1; % flip diagonally       (dimensions flip)
                                if (flip_flag == 1)
                                    sub_image = (rot90(flipdim(sub_image,2),1)); 
                                end
                                %flip_flag = 2; % rotate by 90 degrees  (dimensions flip)
                                if (flip_flag == 2)
                                    sub_image = rot90(sub_image,1);  
                                end
                                %flip_flag = 3; % rotate by 180 degrees (dimensions remain the same)
                                if (flip_flag == 3)
                                    sub_image = rot90(sub_image,2);    
                                end
                                %flip_flag = 4; % rotate by 270 degrees (dimensions flip)
                                if (flip_flag == 4)
                                    sub_image = rot90(sub_image,3);  
                                end
                                %flip_flag = 5; % flip up down       (dimensions remain the same)
                                if (flip_flag == 5)
                                    sub_image = flipdim(sub_image,1); 
                                end
                                %flip_flag = 6; % flip left right    (dimensions remain the same)
                                if (flip_flag == 6)
                                    sub_image = flipdim(sub_image,2); 
                                end
                                Patient{i}.Study{j}.Series{k}.Total_Images = Patient{i}.Study{j}.Series{k}.Total_Images + 1;
                                
                                maximum_pixel = max(max(sub_image));
                                if (maximum_pixel == 0)
                                    % this image is one of the padded images at the end, therefore don't output it
                                    break;  
                                else
                                    A(counter: counter+(dim(2)*dim(3))-1) = reshape(sub_image,dim(2)*dim(3),1);
                                    counter = counter + dim(2)*dim(3);
                                    
                                    image_output_counter = image_output_counter + 1;
                                end
                                
                            end
                        end
                        dim(4) = image_output_counter;
                        
                        Patient{i}.Study{j}.Series{k}.MatrixDimensions = dim(2:8);    % Put data size (matrix dimensions) into the output storage
                        Patient{i}.Study{j}.Series{k}.VoxelSizes = pixdim(2:8);       % Put the voxel dimensions into the output storage
                        % Strictly this is the sampling spacing
                        %Patient{i}.Study{j}.Series{k}.SliceThickness = 
                        
                        if (tiff_data == 1)
                            %regrid onto a 1mm matrix so that  NIH image can handle the data
                            max_x = dim(2);
                            max_y = dim(3);
                            dim_x = pixdim(2);
                            dim_y = pixdim(3);
                            image_data = reshape(A,max_x,max_y);
                            size(image_data);
                            output_matrix = interp2(dim_y*[0:max_y-1],dim_x*[0:max_x-1]',double(reshape(A,max_x,max_y)),[0:max_y*dim_y-1],[0:max_x*dim_x-1]');
                            imwrite(uint8(output_matrix) , [ images_dir delimiter num2str(l) '.tiff' ] , 'tiff','compression','none', 'Resolution',[25.4 25.4]);
                            
                        end
                        if (analyze_data == 1)
                            analyze_data_type = 'short';
                            
                            
                            local_min = min(min(min(A)));
                            if (local_min < cal_min) 
                                cal_min = local_min;
                            end
                            local_max = max(max(max(A)));
                            if (local_max > cal_max) 
                                cal_max = local_max;
                            end
                            
                            fwrite(fp,A,analyze_data_type);
                            
                            data_output = 1;
                        end
                    end
                end
            end
        end
    end 
    if (flag == 1) 
        % Only one image in the dataset
        %dim(4) = 1; %this was a bug in the version that stuffed up when only a single mosaic file was present 
        pixdim(4) = 1;
    end
else
    % This isn't MOSAIC data so we can just dump it
    
    %Assume all images are the same size
    flag = 0;
    counter = 1;
    image_counter = 0;
    for i = patient_no
        for j = study
            for k = series
                
                Patient{i}.Study{j}.Series{k}.Total_Images = 0;
                for l = image
                    if ( isstruct (Patient{i}.Study{j}.Series{k}.Image{l}) == 1)
                        image_counter = image_counter + 1;
                        %Only do anything at all if there is data present!
                        fprintf('.');
                        if (flag == 1)
                            %Now I have the second slice I can determine the slice to slice distance (i.e. z_pixel dimension)
                            %Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient
                            %z_slice_posn
                            pixdim(4) = magnitude(Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient - z_slice_posn);
                            slice_vector = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient - z_slice_posn;
                            
                            flag = 2;    
                        end
                        if (flag == 0)
                            %Data dimensions
                            dim = zeros(8,1);
                            
                            dim(1) = 4; % number of dimensions
                            dim(2) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.Rows;
                            dim(3) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.Columns;
                            %dim(4) = length(patient_no) * length(study) * length(series) * length(image);
                            dim(5) = 1; %This has been assumed but we probably need to be cleverer about dim(4) and dim(5)
                            dim(6) = 1; % only 4D therefore irrelevant
                            dim(7) = 1; % only 4D therefore irrelevant
                            dim(8) = 1; % only 4D therefore irrelevant
                            
                            pixdim = zeros(8,1);
                            pixdim(1) = 0;
                            pixdim(2) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.PixelSpacing(1); % these may be the wrong way around
                            pixdim(3) = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.PixelSpacing(2); %these may be the wrong way around
                            %I can only get the z_pixel spacing from the interslice distance (how do I do this for Mosaic?)
                            z_slice_posn = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient;
                            pixdim(5) = 1; %This is arbitrary
                            pixdim(6) = 1; % only 4D therefore irrelevant
                            pixdim(7) = 1; % only 4D therefore irrelevant
                            pixdim(8) = 1; % only 4D therefore irrelevant
                            
                            A = zeros(dim(2)*dim(3),1);
                            
                            if (flip_flag == 1 | flip_flag == 2 | flip_flag == 4)
                                %Swap the dimensions 
                                temp = dim(2);
                                dim(2) = dim(3);
                                dim(3) = temp;
                                % And the voxel sizes
                                temp = pixdim(2);
                                pixdim(2) = pixdim(3);
                                pixdim(3) = temp;
                            end
                            
                            %Snag the slice orientation information for the aux file
                            pixel_origin = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImagePositionPatient;       % 3 floats
                            %Next the vector describing how to get from one slice to the next
                            slice_vector = [0.0 , 0.0 , 0.0 ]; %this will be created when flag == 1 (when we get to the next slice)
                            %Next the in-plane vector for the data as it is in the Dicom file
                            read_vector = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImageOrientationPatient(1:3);% 3 floats for read direction
                            phase_vector  = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.ImageOrientationPatient(4:6);       % 3 floats for phase encode direction
                            %This describes the orientation of the patient within the magnet
                            Patient_Position = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.PatientPosition;
                            %This is probable extra information, but might be useful
                            %slice_location = Patient{i}.Study{j}.Series{k}.Image{l}.Whole_Structure.SliceLocation;
                            %Finally the information concerning whether the images have been flipped in the output file
                            %flip_flag
                            
                            
                            flag = 1;                    
                        end
                        
                        %
                        % This is to copy the file to the same directory as the output data
                        % therefore, trim off the directory part of the filename
                        full_filename = char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename);
                        
                        slashes = strfind( full_filename , delimiter);
                        
                        filename_bit = full_filename (slashes(length(slashes)) : length(full_filename));
                        
                        if (dicom_copy == 1)    
                            %Do the normal copy (if TPM then do an additional copy)
                            copyfile( char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename) , [ images_dir delimiter filename_bit ] , 'writable');
                            
                            
                        end
                        
                        if (tpm_file == 1 | lategd_file == 1)
                            if (isempty(strmatch('Whole_Structure',fieldnames(Patient{i}))))
                                % Create a WHole_Structure for the patient
                                Patient{i}.Whole_Structure.PatientID = Patient{i}.PatientID; %This does seem to be mindless duplication!
                                Patient{i}.Whole_Structure.PatientsName.FamilyName = 'not_recorded';
                                Patient{i}.Whole_Structure.PatientsName.GivenName = 'not_recorded';
                            end
                            if (isempty(strmatch('Whole_Structure',fieldnames(Patient{i}.Study{j}))))
                                Patient{i}.Study{j}.Whole_Structure.StudyDate = 'not_recorded';
                            end        
                        end
                        
                        Patient{i}.Whole_Structure.PatientID = legitimise_filename(Patient{i}.Whole_Structure.PatientID);
                        
                        if (isempty(strfind(Patient{i}.Study{j}.Series{k}.Whole_Structure.ProtocolName,'TPM')) & isempty(strfind(Patient{i}.Study{j}.Series{k}.Whole_Structure.ProtocolName,'tpm')))
                            % This is not a Tissue Phase Map file so carry on as normal           
                        else
                            if (tpm_file == 1)
                                %The Freiburg TPM code requires that '.dcm' is the suffix on all file names
                                % Also put the Frieburg TPM datafiles into a single TPM directory
                                TPM_dir = [output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)  delimiter 'TPM' ];
                                if (isequal(exist(TPM_dir,'dir'),0) )
                                    mkdir([output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)  delimiter] , 'TPM');
                                end
                                %What should the output filename be?
                                % 
                                tmp_series_name = sprintf('00000%d',k);
                                tmp_image_name  = sprintf('00000%d',l);
                                try
                                TPM_filename = [Patient{i}.Whole_Structure.PatientsName.FamilyName '_' Patient{i}.Whole_Structure.PatientsName.GivenName '__' ...
                                        '_' Patient{i}.Study{j}.Whole_Structure.StudyDate ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                            catch
                                 TPM_filename = ['not_recorded' ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                          
                                %MiddleName removed as it crashes when there is no middle name!
                            end
                                
                                
                                copyfile( char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename) , [ TPM_dir delimiter TPM_filename] , 'writable');
                            end
                        end
                        if (isempty(strfind(lower(Patient{i}.Study{j}.Series{k}.Whole_Structure.ProtocolName),lower(lategd_filename))))
                            % This is not a Tissue Phase Map file so carry on as normal           
                        else
                            if (lategd_file == 1)
                                %The Freiburg TPM code requires that '.dcm' is the suffix on all file names
                                % Also put the Frieburg TPM datafiles into a single TPM directory
                                LATEGD_dir = [output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)  delimiter lategd_filename ];
                                if (isequal(exist(LATEGD_dir,'dir'),0) )
                                    mkdir([output_pathname 'Patient_' Patient{i}.Whole_Structure.PatientID delimiter 'Study_' num2str(j)] , lategd_filename );
                                end
                                %What should the output filename be?
                                % 
                                tmp_series_name = sprintf('00000%d',k);
                                tmp_image_name  = sprintf('00000%d',l);
                                try
                                LATEGD_filename = [Patient{i}.Whole_Structure.PatientsName.FamilyName '_' Patient{i}.Whole_Structure.PatientsName.GivenName '__' ...
                                        '_' Patient{i}.Study{j}.Whole_Structure.StudyDate ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                            catch
                                LATEGD_filename = ['not_recorded' ...
                                        '_s' tmp_series_name((-2:0)+length(tmp_series_name)) '_i' tmp_image_name((-3:0)+length(tmp_image_name)) '.dcm'];
                        
                                %MiddleName removed as it crashes when there is no middle name!
                            end
                                
                                
                                copyfile( char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename) , [ LATEGD_dir delimiter LATEGD_filename] , 'writable');
                            end
                        end
                        
                        % Make the output name that we use here relative to the location of the Patient.mat file
                        relative_name = images_dir(length(output_pathname)+1:length(images_dir));
                        Patient{i}.Study{j}.Series{k}.Image{l}.stored_dir =  [ relative_name delimiter filename_bit ];
                        
                        dim(4) = image_counter ; %This is necessary as the 'image' array may contatin empty elements
                        
                        
                        
                        % Open the dicomfile
                        tmp_image = dicomread(char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename));
                        
                        %
                        % Check the data is what we expect
                        if (ndims(tmp_image) ~= 2)
                            % Data is not 2 dimensional 
                            fprintf('Data is not 2 dimensional!\n');
                        end
                        if ( isa(tmp_image , 'uint16') == 0)
                            % Data is not unsigned integer with 16bits
                            fprintf('Data is not 16bit unsigned integer\n');
                        end
                        
                        sub_image = tmp_image;
                        
                        %flip_flag = 0; % Don't flip            (dimensions remain the same)
                        
                        %flip_flag = 1; % flip diagonally       (dimensions flip)
                        if (flip_flag == 1)
                            sub_image = (rot90(flipdim(sub_image,2),1)); 
                        end
                        %flip_flag = 2; % rotate by 90 degrees  (dimensions flip)
                        if (flip_flag == 2)
                            sub_image = rot90(sub_image,1);  
                        end
                        %flip_flag = 3; % rotate by 180 degrees (dimensions remain the same)
                        if (flip_flag == 3)
                            sub_image = rot90(sub_image,2);    
                        end
                        %flip_flag = 4; % rotate by 270 degrees (dimensions flip)
                        if (flip_flag == 4)
                            sub_image = rot90(sub_image,3);  
                        end
                        %flip_flag = 5; % flip up down       (dimensions remain the same)
                        if (flip_flag == 5)
                            sub_image = flipdim(sub_image,1); 
                        end
                        %flip_flag = 6; % flip left right    (dimensions remain the same)
                        if (flip_flag == 6)
                            sub_image = flipdim(sub_image,2); 
                        end
                        
                        try
                            A = reshape(sub_image,dim(2)*dim(3),1);
                            Patient{i}.Study{j}.Series{k}.Total_Images = Patient{i}.Study{j}.Series{k}.Total_Images + 1;
                        catch
                            A = zeros(dim(2)*dim(3),1);
                            disp2(['Failed to reshape '  , char(Patient{i}.Study{j}.Series{k}.Image{l}.Filename) ,  '. This means that this image is different in size to the first image in the series. \n Probably Argus file or something equally ugly. Replace with zeros and carry on.']);
                        end
                        
                        Patient{i}.Study{j}.Series{k}.MatrixDimensions = dim(2:8);    % Put data size (matrix dimensions) into the output storage
                        Patient{i}.Study{j}.Series{k}.VoxelSizes = pixdim(2:8);       % Put the voxel dimensions into the output storage
                        % Strictly this is the sampling spacing
                        %Patient{i}.Study{j}.Series{k}.SliceThickness = 
                        
                        if (tiff_data == 1)
                            %regrid onto a 1mm matrix so that  NIH image can handle the data
                            max_x = dim(2);
                            max_y = dim(3);
                            dim_x = pixdim(2);
                            dim_y = pixdim(3);
                            image_data = reshape(A,max_x,max_y);
                            size(image_data);
                            output_matrix = interp2(dim_y*[0:max_y-1],dim_x*[0:max_x-1]',double(reshape(A,max_x,max_y)),[0:max_y*dim_y-1],[0:max_x*dim_x-1]');
                            imwrite(uint8(output_matrix) , [ images_dir delimiter num2str(l) '.tiff' ] , 'tiff','compression','none', 'Resolution',[25.4 25.4]);
                            
                        end
                        
                        if (analyze_data == 1)
                            
                            local_min = min(min(min(A)));
                            if (local_min < cal_min) 
                                cal_min = local_min;
                            end
                            local_max = max(max(max(A)));
                            if (local_max > cal_max) 
                                cal_max = local_max;
                            end
                            
                            analyze_data_type = 'short';
                            
                            fwrite(fp,A,analyze_data_type);
                            data_output = 1;
                        end
                        counter = counter + dim(2)*dim(3);
                    end
                end
            end
        end
    end  
    if (flag == 1) 
        % Only one image in the dataset
        dim(4) = 1;
        pixdim(4) = 1;
    end
end

if (analyze_data == 1)
    fclose(fp);
end

if (data_output == 0)
    % No data was output, so delete the  .img file
    %disp2(['Deleting ' , output_filename ]);
    %delete([output_filename '.img']);
else
    % Now output a header for this data
    header_filename    		= [output_filename '.hdr'];
    
    % For byte swapped data-types, also swap the bytes around in the headers.
    mach = 'native';
    %if spm_type(TYPE,'swapped'),
    %	if spm_platform('bigend'),
    %		mach = 'ieee-le';
    %	else,
    %		mach = 'ieee-be';
    %	end;
    %	TYPE = spm_type(spm_type(TYPE));
    %end;
    fid             = fopen(header_filename,'w',mach);
    
    if (fid == -1),
        error(['Error opening ' header_filename '. Check that you have write permission.']);
    end;
    %---------------------------------------------------------------------------
    data_type 	= ['dsr      ' 0]; %Leave it as this!
    
    
    % set header variables
    %---------------------------------------------------------------------------
    %DIM		= DIM(:)'; if size(DIM,2) < 4; DIM = [DIM 1]; end  %This is the dimensions of the voxel data from SPM
    %VOX		= VOX(:)'; if size(VOX,2) < 4; VOX = [VOX 0]; end  %This is the Voxel data in SPM
    %dim		= [4 DIM(1:4) 0 0 0];	
    %pixdim		= [0 VOX(1:4) 0 0 0];
    vox_offset      = 0; % spm uses OFFSET;
    funused1	= 1;  % spm uses SCALE;
    bitpix 		= 0;
    descrip         = zeros(1,80);
    aux_file        = ['none                   ' 0];
    %The following removes the pathname from the filename, hopefully bringing it down to 24characters!
    delims = strfind(output_filename,delimiter);
    mdr_aux_file = [ output_filename( delims(length(delims))+1 : length(output_filename))  '.mdr'  ];
    aux_output_pathname = [output_filename(1:delims(length(delims)))];
    if (length(mdr_aux_file) > 24)
        mdr_aux_file = [mdr_aux_file(1:19) '.mdr' 0];
    else
        mdr_aux_file = [mdr_aux_file '                            '];
        mdr_aux_file        = [mdr_aux_file(1:23) 0];
    end
    
    aux_file = mdr_aux_file;
    
    origin          = [0 0 0 0 0];
    
    TYPE = 4; %This seems to be the format of Siemens Data
    %---------------------------------------------------------------------------
    if TYPE == 1;   bitpix = 1;   glmax = 1;        glmin = 0;	end
    if TYPE == 2;   bitpix = 8;   glmax = 255;      glmin = 0;	end
    if TYPE == 4;   bitpix = 16;  glmax = 32767;    glmin = 0;  	end
    if TYPE == 8;   bitpix = 32;  glmax = (2^31-1); glmin = 0;	end
    if TYPE == 16;  bitpix = 32;  glmax = 1;        glmin = 0;	end
    if TYPE == 64;  bitpix = 64;  glmax = 1;        glmin = 0;	end
    
    %---------------------------------------------------------------------------
    
    data_type = 'OCMR DICOM';
    
    tmp_db_name = [Patient{patient_no(1)}.PatientID ' ' num2str(study(1)) ' ' num2str(series(1)) ' ' num2str(image(1)) '                        '];
    db_name   = tmp_db_name(1:18);
    
    
    fseek(fid,0,'bof');
    
    % write (struct) header_key
    %---------------------------------------------------------------------------
    fwrite(fid,348,		'int32');       %size of header
    fwrite(fid,data_type,	'char' );   %data_type[10]
    fwrite(fid,db_name,	'char' );       %db_name[18]
    fwrite(fid,0,		'int32');       %extents
    fwrite(fid,0,		'int16');       %session_error
    fwrite(fid,'r',		'char' );       %regular
    fwrite(fid,'0',		'char' );       %hkey_un0
    
    % write (struct) image_dimension
    %---------------------------------------------------------------------------
    fseek(fid,40,'bof');
    
    %disp2(pixdim');
    
    fwrite(fid,dim,		'int16');       %dim[8]
    fwrite(fid,'mm',	'char' );       %unused8
    fwrite(fid,0,		'char' );       %unused9
    fwrite(fid,0,		'char' );       %unused9
    
    fwrite(fid,zeros(1,8),	'char' );   %unused 10 , 11 , 12 , 13 
    fwrite(fid,0,		'int16');       %unused14
    fwrite(fid,TYPE,	'int16');       %datatype
    fwrite(fid,bitpix,	'int16');       %bitpix
    fwrite(fid,0,		'int16');       %dim_un0
    fwrite(fid,pixdim,	'float');       %pix_dim[8], width, height, thickness etc.
    fwrite(fid,vox_offset,	'float');   %vox_offset
    fwrite(fid,funused1,	'float');   %funused1 SPM uses this for SCALE, we can copy this
    fwrite(fid,0,		'float');       %funused2  USE THIS FOR b-factor
    fwrite(fid,0,		'float');       %funused3
    fwrite(fid,cal_max,		'float');       %cal_max (float)
    fwrite(fid,cal_min,		'float');       %cal_min   (float)
    fwrite(fid,0,		'int32');       %compressed (float)
    fwrite(fid,0,		'int32');       %verified (float)
    fwrite(fid,glmax,	'int32');       %glmax
    fwrite(fid,glmin,	'int32');       %glmin
    
    % write (struct) image_dimension
    %---------------------------------------------------------------------------
    fwrite(fid,descrip,	'char');
    fwrite(fid,aux_file,    'char');
    fwrite(fid,0,           'char');
    fwrite(fid,origin,      'int16');
    if fwrite(fid,zeros(1,85), 'char')~=85
        fclose(fid);
        error(['Error writing. Check your disk space.']);
    end
    
    s   = ftell(fid);
    fclose(fid);
    
    %Now write out the auxilliary file.  This should contain information on slice locations and orientations
    fid             = fopen([aux_output_pathname aux_file],'w',mach);
    
    if (fid == -1),
        error(['Error opening ' [aux_output_pathname aux_file] '. Check that you have write permission.']);
    end;
    
    %First the slice offset information, for the first slice
    fwrite(fid,pixel_origin,		'float');       % 3 floats
    %Next the vector describing how to get from one slice to the next
    fwrite(fid,slice_vector,		'float');       % 3 floats
    %Next the in-plane vector for the data as it is in the Dicom file
    fwrite(fid,read_vector,		'float');       % 3 floats for read direction
    fwrite(fid,phase_vector,		'float');       % 3 floats for phase encode direction
    %Next include the patient orientation information i.e.
    % HFS (head first suppine), HFP (head first prone), FFS (feet first supine) etc.
    fwrite(fid,Patient_Position, 'char'); %3 char
    %Finally the information concerning whether the images have been flipped in the output file
    fwrite(fid,flip_flag,		'int16');       % 1 int (as described above)
    fclose(fid);
    
end

return;
%_______________________________________________________________________


function magn = magnitude (vector)
total = 0.0;
for i = 1 : length(vector)
    total = total + vector(i)*vector(i);
end
magn = sqrt(total);
return

function disp2 (text)
% This function displays the text onthe screen but also puts it into the window on the gui
global gui_flag gui_handle
disp(text)

if (gui_flag == 1)
    set(gui_handle,'String',[get(gui_handle,'String') text]);
    pause(0.00001);
end

function parsed_name = legitimise_filename (input_filename)
% Check the Patient ID for dodgy characters and replace them, also change the PatientID in the Patient structure as the same time
parsed_name = input_filename;        
colons = strfind(parsed_name, ':');
if (isempty(colons))
    % Patient_ID does not contain any ':'
else
    parsed_name(colons) = '-';
    disp2(['Colons found while running legitimise_filename, therefore these have been swapped for ;']);
end
spaces = strfind(parsed_name, ' ');
if (isempty(spaces))
    % Patient_ID does not contain any spaces
else
    parsed_name(spaces) = '_';
    disp2(['Spaces found while running legitimise_filename, therefore these have been swapped for _']);
end

return


