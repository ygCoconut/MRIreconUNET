function Profile_tool(varargin);
% function Profile_tool(varargin);
% Function to create and manipulate profiles on a montage of images.
% Use with imagescn or iamgesc.
%
% Usage: Profile_tool;
%
% Author: Daniel Herzka  herzkad@nih.gov
% Laboratory of Cardiac Energetics 
% National Heart, Lung and Blood Institute, NIH, DHHS
% Bethesda, MD 20892
% and 
% Medical Imaging Laboratory
% Department of Biomedical Engineering
% Johns Hopkins University Schoold of Medicine
% Baltimore, MD 21205

if isempty(varargin) 
   Action = 'New';
   Image_handles = [];
else
    if ischar(varargin{1})   % sent in an action
        Action = varargin{1};  
    elseif isnumeric(varargin{1}) % sent in a matrix of handles  
        Action = 'New';
        Image_handles = varargin{1};
    else                     % sent in unidentified material
        Action = 'Exit';
    end
end

% set or clear the global debug flag
%global DB; DB = 1;

switch Action
	
case 'New'
    Create_New_Button(Image_handles);

case 'Activate_Profile_Tool'
    Activate_Profile_Tool(varargin(2:end));
    
case 'Deactivate_Profile_Tool'
    Deactivate_Profile_Tool(varargin{2:end});
 
case 'Create_New_Profile'
    Create_New_Profile;

case 'Delete_Profile'
    Delete_Profile;
    
case 'Copy_Current_Profile'
    Copy_Current_Profile;
	
case 'Paste_Current_Profile'
    Paste_Current_Profile;

case 'Change_Current_Axes'
    Change_Current_Axes;

case 'Change_Current_Profile'
    Change_Current_Profile;

case 'Profile_Size_Adjust_Entry'
    % Entry
    Profile_Size_Adjust_Entry(varargin{2});
case 'Profile_Size_Adjust'
    % Cycle
    Profile_Size_Adjust(varargin{2});
case 'Profile_Size_Adjust_Exit'
    % Exit
    set(gcf, 'WindowButtonMotionFcn', ' ','WindowButtonUpFcn', ' ');
    Profile_Size_Adjust_Exit;    

case 'Profile_Pos_Adjust_Entry'
    % Entry
    Profile_Pos_Adjust_Entry(varargin{2});
case 'Profile_Pos_Adjust'
    % Cycle
    Profile_Pos_Adjust(varargin{2});
case 'Profile_Pos_Adjust_Exit'
    % Exit
    set(gcf, 'WindowButtonMotionFcn', ' ','WindowButtonUpFcn', ' ');
    Profile_Pos_Adjust_Exit;  
	
case 'Resort_Profile_Info_Listbox'
    Resort_Profile_Info_Listbox(varargin{2});
       
case 'Listbox_Change_Current_Profile'
    Listbox_Change_Current_Profile;
case 'Toggle_Axes_View'
    Toggle_Axes_View(varargin{2:end});
case 'Toggle_Profile_View'
    Toggle_Profile_View(varargin{2:end});

	
	
case 'Auto_Axes_Limits'
    Auto_Axes_Limits(varargin{2:end});
case 'Fix_Axes_Limits'
    Fix_Axes_Limits(varargin{2:end});
case 'Toggle_FFT_View'
    Toggle_FFT_View;

    
case 'Save_Profile'
    Save_Profile(varargin{2:end});
case 'Load_Profile'
    Load_Profile(varargin{2:end});
	
case 'Close_Parent_Figure'
    Close_Parent_Figure;   
case 'Menu_Profile_Tool'
	Menu_Profile_Tool;
	
case 'Exit';
    disp('Unknown Input Argument');
    
otherwise
    disp(['Unimplemented Functionality: ', Action]);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Create_New_Button(varargin)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

fig = gcf;

% Find handle for current image toolbar and menubar
hToolbar = findall(fig, 'type', 'uitoolbar', 'Tag','FigureToolBar' );
hToolMenu = findall(fig, 'Label', '&Tools');

if ~isempty(hToolbar) & isempty(findobj(hToolbar, 'Tag', 'figProfileTool'))
	hToolbar_Children = get(hToolbar, 'Children');
    
   % The default button size is 15 x 16 x 3. Create Button Image
   button_size_x= 16;
   button_image = NaN* zeros(15,button_size_x);
   f = [...
           2     3     4     5    17    20    32    35    47    48  ...
           49    50    66    82    98   114   130   146   162   163  ...
           164   165   177   180   192   195   207   208   209   210 ];

   button_image(f) = 0;
   button_image = repmat(button_image, [1,1,3]);
   
   buttontags = {'figWindowLevel', 'figPanZoom', 'figROITool', 'figViewImages', 'figPointTool', 'figRotateTool', 'figProfileTool'};
   separator = 'off';
   
   hbuttons = [];
   for i = 1:length(buttontags)
       hbuttons = [hbuttons, findobj(hToolbar_Children, 'Tag', buttontags{i})];
   end;
   if isempty(hbuttons)
       separator = 'on';
   end;
         
   hNewButton = uitoggletool(hToolbar);
   set(hNewButton, 'Cdata', button_image, ...
      'OnCallback', 'Profile_tool(''Activate_Profile_Tool'')',...
      'OffCallback', 'Profile_tool(''Deactivate_Profile_Tool'')',...
      'Tag', 'figProfileTool', ...
      'TooltipString', 'Create and Manipulate Profiles',...
      'Separator', separator, ...
      'Enable', 'on');   
end;

% If the menubar exists, create menu item
if ~isempty(hToolMenu) & isempty(findobj(hToolMenu, 'Tag', 'menuProfileTool'))
  hWindowLevelMenu = findobj(hToolMenu, 'Tag', 'menuWindowLevel');
  hPanZoomMenu     = findobj(hToolMenu, 'Tag', 'menuPanZoom');
  hROIToolMenu     = findobj(hToolMenu, 'Tag', 'menuROITool');
  hViewImageMenu   = findobj(hToolMenu, 'Tag', 'menuViewImages');
  hPointToolMenu   = findobj(hToolMenu, 'Tag', 'menuPointTool');
  hRotateToolMenu  = findobj(hToolMenu, 'Tag', 'menuRotateTool');
  hProfileToolMenu = findobj(hToolMenu, 'Tag', 'menuProfileTool');
  
  position = 9;
  separator = 'On';
  hMenus = [ hWindowLevelMenu, hPanZoomMenu, hViewImageMenu, hPointToolMenu,hRotateToolMenu ];
  if length(hMenus>0) 
	  position = position + length(hMenus);
	  separator = 'Off';
  end;
  
  hNewMenu = uimenu(hToolMenu,'Position', position);
  set(hNewMenu, 'Tag', 'menuProfileTool','Label',...
      'Profile Tool',...
      'CallBack', 'Profile_tool(''Menu_Profile_tool'')',...
      'Separator', separator,...
      'UserData', hNewButton...
  ); 
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Activate_Profile_Tool(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
if nargin == 0 | isempty(varargin{1})
    set(0, 'ShowHiddenHandles', 'On');
    hNewButton = gcbo;
    set(findobj('Tag', 'menuProfileTool'),'checked', 'on');
else % sent in something (could be empty)
    %if  isempty(varargin{1}(1))
    t = varargin{1};
    hNewButton = t{1};
end;

% allows for calls from buttons other than those in toolbar
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure')
    fig = get(fig, 'Parent');
end;

% Deactivate zoom and rotate buttons
hToolbar = findall(fig, 'type', 'uitoolbar');
hToolbar = findobj(hToolbar, 'Tag', 'FigureToolBar');
if ~isempty(hToolbar)
	hToolbar_Children = get(hToolbar, 'Children');
	
	% disable MATLAB's own tools
	Rot3D = findobj(hToolbar_Children,'Tag', 'figToolRotate3D');
	ZoomO = findobj(hToolbar_Children,'Tag', 'figToolZoomOut');
	ZoomI = findobj(hToolbar_Children,'Tag', 'figToolZoomIn');

	% try to disable other tools buttons - if they exist
	WL = findobj(hToolbar_Children, 'Tag', 'figWindowLevel');
	PZ = findobj(hToolbar_Children, 'Tag', 'figPanZoom');
	RT = findobj(hToolbar_Children, 'Tag', 'figROITool');
	MV = findobj(hToolbar_Children, 'Tag', 'figViewImages');
	PM = findobj(hToolbar_Children, 'Tag', 'figPointTool');
	RotT = findobj(hToolbar_Children, 'Tag', 'figRotateTool');
	Prof = findobj(hToolbar_Children, 'Tag', 'figProfileTool');
	
	old_ToolHandles  =     [Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,PM,RotT];
	old_ToolEnables  = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,PM,RotT], 'Enable');
	old_ToolStates   = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,PM,RotT], 'State');
	
	for i = 1:length(old_ToolHandles)
		if strcmp(old_ToolStates(i) , 'on')			
			set(old_ToolHandles(i), 'State', 'Off');
		end;
		set(old_ToolHandles(i), 'Enable', 'Off');
	end;
        %LFG
        %enable save_prefs tool button
        SP = findobj(hToolbar_Children, 'Tag', 'figSavePrefsTool');
        set(SP,'Enable','On');
end;

% Start ROI Tool GUI
% close the other ROI Tool figuresto avoid conflicts
Close_Old_Figure('Prof_T_figure');  % close old profile tool figures
Close_Old_Figure('Prof_I_figure');  % close old profile info figures
Close_Old_Figure('Prof_P_figure');  % close old profile plot figures

% open new figure
fig_tool_file = 'Profile_tool_figure.fig';
fig_tool = openfig(fig_tool_file,'reuse');
optional_uicontrols = { ...
    'Link_Profile_togglebutton',         'Value'; ...
    'Create_Profile_checkbox',           'Value'; ...
    'Create_Profile_popupmenu',          'Value'; ...
    'Paste_All_checkbox',                'Value'; ...
    'Paste_New_checkbox',                'Value'; ...
    'Delete_Profile_popupmenu',          'Value'; ...
    'Save_MAT_checkbox',                 'Value'; ...
    'Sort_Profile_togglebutton',         'Value'; ...
    'Sort_Image_togglebutton',           'Value'; ...
    'All_Profiles_View_togglebutton',    'Value'; ...
    'Active_Profiles_View_togglebutton', 'Value'; ...
    'Single_Axes_View_togglebutton',     'Value'; ...
    'Multiple_Axes_View_togglebutton',   'Value'; ...
    'Auto_X_radiobutton',                'Value'; ...
    'Min_X_edit',                        'String'; ...
    'Max_X_edit',                        'String'; ...
    'Auto_Y_radiobutton',                'Value'; ...
    'Min_Y_edit',                        'String'; ...
    'Max_Y_edit',                        'String'; ...
                   };
set(SP,'Userdata',{fig_tool, fig_tool_file, optional_uicontrols});

% Generate a structure of handles to pass to callbacks, and store it. 
handlesTool = guihandles(fig_tool);

% Make and set the close String for the ROI Tool figure
close_str = [ 'hNewButton = findobj(''Tag'', ''figProfileTool'');' ...
        ' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
        ' set(hNewButton, ''State'', ''off'' );' ...
        ' else,  ' ...
        ' Profile_tool(''Deactivate_Profile_Tool'' ,hNewButton);'...
        ' set(hNewButton, ''Value'', 0);',...
        ' end;' ];
set(fig_tool, 'Name', 'Profile Tool', 'CloseRequestfcn', close_str);

% Record and store previous WBDF etc to restore state after RT is done. 
old_WBDF = get(fig, 'WindowButtonDownFcn');
old_WBMF = get(fig, 'WindowButtonMotionFcn');
old_WBUF = get(fig, 'WindowButtonUpFcn');
old_UserData = get(fig, 'UserData');
old_CRF = get(fig, 'Closerequestfcn');

% Store initial state of all axes in current figure for reset
% Note that using appdata for storage means that if figure is saved, 
% ROI information is saved and usable too
if isappdata(fig, 'Profile_Info_Table')
	handlesTool.Profile_info_table  = getappdata(fig, 'Profile_Info_Table'  );
	handlesTool.h_all_axes          = getappdata(fig, 'Profile_Axes_Handles');
	handlesTool.Current_Profile     = getappdata(fig, 'Profile_Current_Info');
    handlesTool.Current_Axes        = getappdata(fig, 'Profile_Current_Axes');
else
	handlesTool.Profile_info_table = Init_Profile_info_table;
	handlesTool.h_all_axes = Find_All_Axes(fig);
	% Pick current axis as top left axes
	handlesTool.Current_Axes = handlesTool.h_all_axes(1,1);
	handlesTool.Current_Profile = [];
end;
handlesTool.colororder = repmat('rbgymcw',1,2);

% store all the old nextplots and bdf's
h_all_axes=handlesTool.h_all_axes';

for i = 1:length(find(h_all_axes(:)))
    if (h_all_axes(i))
		 old_axes_BDF{i} = get(h_all_axes(i), 'ButtonDownFcn');
		 old_axes_NextPlot{i} = get(h_all_axes(i),'NextPlot');
		 h_image = findobj(h_all_axes(i), 'Type', 'Image');
		 
		 set(h_all_axes(i),'NextPlot', 'add');
		 old_image_BDF{i} = get(h_image, 'ButtonDownFcn');
		 set(h_image,'ButtonDownFcn', 'Profile_tool(''Change_Current_Axes'')');
		 if isappdata(h_all_axes(i), 'ImageData');
			 handlesTool.Image_Data{i} = getappdata(h_all_axes(i), 'ImageData');
		 else
			 handlesTool.Image_Data{i} = get(findobj(h_all_axes(i), 'Type', 'Image'), 'Cdata');
		 end;
		 
    end;
end;

% Draw faster and without flashes
set(fig, 'Closerequestfcn', [ old_CRF , ',Profile_tool(''Close_Parent_Figure'')']);
set(fig, 'Renderer', 'zbuffer');
set(fig, 'CurrentAxes', handlesTool.Current_Axes);
set(0,   'ShowHiddenHandles', 'On', 'CurrentFigure', fig);
set(handlesTool.h_all_axes(find(handlesTool.h_all_axes)), 'Drawmode', 'Fast');

% store the figure's old infor within the fig's own userdata
set(fig, 'UserData', {fig_tool, old_WBDF, old_WBMF, old_WBUF, ... 
		old_UserData, old_axes_BDF, old_axes_NextPlot, old_CRF, old_image_BDF, ...
		old_ToolEnables,old_ToolHandles});

% Now check if previous use of Profile Tool left a Profile_info_table with handles to reset
if isempty([handlesTool.Profile_info_table.Profile_Exists])
    % disable all buttons until an ROI has been created
    Change_Object_Enable_State(handlesTool, 'Off', 1);
    handlesTool.Info_figure = [];
    handlesTool.Plot_figure = [];
    
    % store all relevant info for faster use during calls
    handlesTool.Tool_figure = fig_tool;
    handlesTool.Parent_figure = fig;
    % Store handles back into the gui
    guidata(handlesTool.Tool_figure,handlesTool);
    
else    
    % set all graphics related to each Profile to visible and allot the appropiate callback to each one    
	% Loop over as temporal and 2d Profiles are composed of different
	% number of elements
	for i = 1:length(handlesTool.Profile_info_table(:))
		if ~isempty(handlesTool.Profile_info_table(i).Profile_Exists)
			Profile_Elements = [handlesTool.Profile_info_table(i).Profile_Elements];
			if handlesTool.Profile_info_table(i).Profile_Type~=4
				% 2D Profile
				%h_line, h_end1, h_end2, h_center, h_number; 
				set(Profile_Elements(1),'Visible', 'On', 'ButtonDownFcn', 'Profile_tool(''Change_Current_Profile'')');    
				set(Profile_Elements(2),'Visible', 'On', 'ButtonDownFcn', 'Profile_tool(''Profile_Size_Adjust_Entry'',1)');    
				set(Profile_Elements(3),'Visible', 'On', 'ButtonDownFcn', 'Profile_tool(''Profile_Size_Adjust_Entry'',2)');    
				set(Profile_Elements(4),'Visible', 'On', 'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'',1)');    
				set(Profile_Elements(5),'Visible', 'On', 'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'',2)');    
			else
				% Temporal Profile
				% h_center, h_number
				set(Profile_Elements(1),'Visible', 'On', 'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'',1)');    
				set(Profile_Elements(2),'Visible', 'On', 'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'',2)');    
			end;
		end;
	end;
				
    % Enable Buttons since ROIs exists, but not the paste objects
    Change_Object_Enable_State(handlesTool, 'Off', 1);
    Change_Object_Enable_State(handlesTool, 'On', 0);
    
    % open new figure for Profile information and draw the Profiles
    handlesTool.Tool_figure = fig_tool;
    handlesTool.Parent_figure = fig;
    guidata(fig_tool, handlesTool);
    Create_Profile_Info_Figure;
    Create_Profile_Plot_Figure;

    % now remeasure all data, update the string table,
    % post the string table and highlight the current profile
    Update_Profile_Info;
    Update_Profile_Info_String;
    Resort_Profile_Info_Listbox;
    Highlight_Current_Profile(handlesTool.Current_Profile);
    % No need to update the plot; creation of the plot figure 
    % automatically does this
    %    Update_Profile_Plot;
    handlesTool = guidata(fig_tool);
    figure(handlesTool.Parent_figure);
    figure(handlesTool.Tool_figure);
    figure(handlesTool.Info_figure);
    figure(handlesTool.Plot_figure);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Deactivate_Profile_Tool(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

if nargin == 0     
    % called from button
    set(0, 'ShowHiddenHandles', 'On');    
    hNewButton = gcbo;
    set(findobj('Tag', 'menuProfileTool'),'checked', 'Off');
else
    % called from menu
    hNewButton = varargin{1};
end;
    
% Reactivate other buttons
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure'),
    fig = get(fig, 'Parent');
end

hToolbar = findall(fig, 'type', 'uitoolbar');
if ~isempty(hToolbar)
    hToolbar_Children = get(hToolbar, 'Children');
    set(findobj(hToolbar_Children,'Tag', 'figToolRotate3D'),'Enable', 'On');
    set(findobj(hToolbar_Children,'Tag', 'figToolZoomOut'),'Enable', 'On');
    set(findobj(hToolbar_Children,'Tag', 'figToolZoomIn'),'Enable', 'On'); 
end;
   
% Restore old WBDFs and USer Data
old_info= get(fig,'UserData');
fig_info = old_info{1};
set(fig, 'WindowButtonDownFcn', old_info{2});
set(fig, 'WindowButtonUpFcn', old_info{3});
set(fig, 'WindowButtonMotionFcn', old_info{4});
set(fig, 'UserData', old_info{5});
set(fig, 'closerequestfcn', old_info{8});
old_ToolEnables  = old_info{10}; 
old_ToolHandles = old_info{11};

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));

% restore axes BDF's and hide all objects by making invisible
% and clearing all object's bdf's
Profile_Exists = [handlesTool.Profile_info_table.Profile_Exists];

%save temp_old_info old_info h_all_axes Userdata
h_all_axes = handlesTool.h_all_axes';   % TEMP
for i = 1:length(find(h_all_axes(:)))
	set(h_all_axes(i),'ButtonDownFcn', char(old_info{6}(i)), 'NextPlot', char(old_info{7}(i)));    
	h_image = findobj(h_all_axes(i), 'Type', 'Image');
	set(h_image, 'ButtonDownFcn', char(old_info{9}(i)));
end     

if ~isempty(Profile_Exists)
	% in case a Profile Exists, make the profiles in the window invisible
	set([handlesTool.Profile_info_table(:).Profile_Elements], 'Visible', 'Off', 'ButtonDownFcn', '');
    % Clear the old line elements (erased upon closing of plot figure)
    [handlesTool.Profile_info_table(:).Single_Line_Elements] = deal([]);
    [handlesTool.Profile_info_table(:).Multiple_Line_Elements] = deal([]);
end;

setappdata(fig, 'Profile_Info_Table',   handlesTool.Profile_info_table);
setappdata(fig, 'Profile_Axes_Handles', handlesTool.h_all_axes);
setappdata(fig, 'Profile_Current_Info', handlesTool.Current_Profile);
setappdata(fig, 'Profile_Current_Axes', handlesTool.Current_Axes);

Close_Old_Figure([],handlesTool.Tool_figure);
Close_Old_Figure([],handlesTool.Info_figure);
Close_Old_Figure([],handlesTool.Plot_figure); 

for i = 1:length(old_ToolHandles)
	try
		set(old_ToolHandles(i), 'Enable', old_ToolEnables{i});
	catch
	end;
end;
%LFG
%disable save_prefs tool button
SP = findobj(hToolbar_Children, 'Tag', 'figSavePrefsTool');
set(SP,'Enable','Off');

set(0, 'ShowHiddenHandles', 'Off');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Create_New_Profile(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function creates new ROI. If this is the first ROI at all, creates the table of 
% ROI info. If the table alreay exists, then adds a new row to the ROI info table
% ROI info table is ROIs x Num Images long;
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;

% load data of interest
fig_tool = findobj('Tag', 'Prof_T_figure');
handlesTool = guidata(fig_tool);

GENERAL = 1; HORIZONTAL = 2; VERTICAL = 3; TEMPORAL = 4;

apply_all = get(handlesTool.Create_Profile_checkbox,'Value');
Create_Profile_method = get(handlesTool.Create_Profile_popupmenu, 'Value');

first_Profile_flag = 0;

% if all Profile exist fields are empty, then this is the first Profile
if isempty([handlesTool.Profile_info_table(:).Profile_Exists])
	% create new Profile_info_table to correct size (rows = number of Profiles, columns = images) 
	% now make the invalid last in case grid is not complete
	%temp_h_all_axes = find(handlesTool.h_all_axes);
	% cycle through to create 0's in the Exist field. Marks the ROI table as initialized
	handlesTool.Profile_info_table = repmat(Init_Profile_info_table, [1, length(find(handlesTool.h_all_axes))]);
    first_Profile_flag = 1;       
end;
Current_Profile_index = Determine_New_Index(handlesTool.Profile_info_table);

if apply_all % create ROI in all images
    % flipud to create ROI in reverse order... so that the current ROI index 
    % one on the first image  
    %t_h_all_axes = handles.Proh_all_axes';
    h_axes_interest = flipud(handlesTool.h_all_axes(find(handlesTool.h_all_axes)));
else
    h_axes_interest = handlesTool.Current_Axes;
end;


switch Create_Profile_method
    case {HORIZONTAL, VERTICAL, GENERAL}
        % create a general 2D profile
        percent_length_Profile = 0.50;
        percent_distance_Number = 0.05;
        for i = 1:length(h_axes_interest(:))
            set(handlesTool.Parent_figure, 'CurrentAxes', h_axes_interest(i));
            h_axes_index = find(handlesTool.h_all_axes'==h_axes_interest(i));
            set(0, 'CurrentFigure', handlesTool.Parent_figure);
            
            xlim = get(gca, 'xlim');
            ylim = get(gca, 'ylim');
            center_x = mean(xlim);
            center_y = mean(ylim);
            
            size_x = diff(xlim)*percent_length_Profile;
            size_y = diff(ylim)*percent_length_Profile;
            
            pos_x1 = center_x - size_x/2; 		pos_x2 = center_x + size_x/2;
            pos_y1 = center_y - size_y/2;	    pos_y2 = center_y + size_y/2;
            
            if (Create_Profile_method==HORIZONTAL) | (Create_Profile_method==GENERAL)
                pos_y1 = center_y;
                pos_y2 = center_y;
                number_distance = diff(xlim)*percent_distance_Number;
            else % VERTICAL 
                pos_x1 = center_x;
                pos_x2 = center_x;
				t = pos_y2; pos_y2 = pos_y1; pos_y1 = t;
                number_distance = diff(ylim)*percent_distance_Number;          
            end;
            [center_x, center_y, number_x, number_y] = Determine_Profile_Elements(pos_x1, pos_y1, pos_x2, pos_y2, number_distance);
            
            % handles_values = [ h_line, h_end1, h_end2, h_center, h_number]		
            handle_values = Make_2D_Profile_Elements(...
                center_x, center_y, ...
                pos_x1, pos_y1, ...
                pos_x2, pos_y2, ...
                number_x, number_y, ...
                handlesTool.colororder(Current_Profile_index), ...
                Current_Profile_index ...
            );
            
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Exists = 1;
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Elements = handle_values;
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Type = Create_Profile_method;
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Color = handlesTool.colororder(Current_Profile_index);
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).N = number_distance;
            
            handlesTool.Current_Profile = [Current_Profile_index, h_axes_index];
            
            set(handle_values, 'Userdata', handlesTool.Current_Profile);
            update_list(i,:) = [Current_Profile_index, h_axes_index];
            guidata(findobj('Tag', 'Prof_T_figure'), handlesTool);
            
            drawnow
		end;		
		% call the Profile info update function: puts data into Profile_info_table
		Update_Profile_Info(update_list);
        
		
	case TEMPORAL
		% Define a temporal profile by placing a single '+' in the middle
		% of the image. If there is not temporal data in the appdata, then
		% simply display the point value
		
		percent_distance_Number = 0.05;
        for i = 1:length(h_axes_interest(:))
            set(handlesTool.Parent_figure, 'CurrentAxes', h_axes_interest(i));
            h_axes_index = find(handlesTool.h_all_axes'==h_axes_interest(i));
            set(0, 'CurrentFigure', handlesTool.Parent_figure);
            
            xlim = get(gca, 'xlim');
            ylim = get(gca, 'ylim');
            center_x = mean(xlim);
            center_y = mean(ylim);            
            
            number_distance = max([diff(xlim), diff(ylim)])* percent_distance_Number / sqrt(2);
            
            number_x = center_x + number_distance;
            number_y = center_y - number_distance;
            
            h_center = plot(center_x, center_y , ...
                ['+', handlesTool.colororder(Current_Profile_index)], ...
                'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'', 1)');
            h_number = text(number_x, number_y, num2str(Current_Profile_index), ...
                'Color', handlesTool.colororder(Current_Profile_index) , ...
                'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'', 2)');            
			
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Exists = 1;
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Elements = [h_center, h_number];
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Type = Create_Profile_method;
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).Profile_Color = handlesTool.colororder(Current_Profile_index);
            handlesTool.Profile_info_table(Current_Profile_index,h_axes_index).N = number_distance;
            
            handlesTool.Current_Profile = [Current_Profile_index, h_axes_index];
            
            set([h_center, h_number], 'Userdata', handlesTool.Current_Profile);
            update_list(i,:) = [Current_Profile_index, h_axes_index];
            guidata(findobj('Tag', 'Prof_T_figure'), handlesTool);
            
            drawnow
            
        end;
end;

% call the Profile info update function: puts data into Profile_info_table
Update_Profile_Info(update_list);

if first_Profile_flag
	% creates figure the first time and creates the string table that is to be
	% used for "publishing" the ROI data
	Create_Profile_Info_Figure;
	Create_Profile_Plot_Figure;
	Change_Object_Enable_State(handlesTool,'Off',1);
	Change_Object_Enable_State(handlesTool,'On',0);
	% Don't need to update string if this is the first ROI
else
	% call function that will take info string table and "publish" it			
	Update_Profile_Info_String(update_list);
end;
% update current ROI index
Resort_Profile_Info_Listbox;
Highlight_Current_Profile(handlesTool.Current_Profile);
Update_Profile_Plot(update_list);
Update_Edit_Boxes;       



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Copy_Current_Profile;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to Copy the current Profile. Stores the Current_Profile
% information in the Copy_Profile_Info field for the Paste function
% to retrieve
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
handlesTool.Copy_Profile_Info = handlesTool.Profile_info_table(handlesTool.Current_Profile(1), handlesTool.Current_Profile(2));
handlesTool.Copy_Profile_Info.Current_Profile = handlesTool.Current_Profile(1);
guidata(findobj('Tag', 'Prof_T_figure'),handlesTool);
set([handlesTool.Paste_Profile_pushbutton, handlesTool.Paste_All_checkbox, handlesTool.Paste_New_checkbox], 'Enable', 'On');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Paste_Current_Profile;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['ROI_Tool: ', Get_Current_Function]); end;

hT = guidata(findobj('Tag', 'Prof_T_figure'));
h_Paste_axes = hT.Current_Axes;
h_Copy_axes  = get(hT.Copy_Profile_Info.Profile_Elements(1), 'Parent');
i_Copy_Profile  = hT.Copy_Profile_Info;

Paste_all = get(hT.Paste_All_checkbox, 'Value');
Paste_new = get(hT.Paste_New_checkbox, 'Value');

if Paste_new
    i_Copy_Profile.Current_Profile = Determine_New_Index(hT.Profile_info_table);
    % Initialize the Current Profiles to mark their existence    
    [hT.Profile_info_table(i_Copy_Profile.Current_Profile,[1:length(find(hT.h_all_axes))]).Profile_Exists] = deal([]);
end;
h_all_axes = hT.h_all_axes';    

indexes_of_interest = [];
if Paste_all & ~Paste_new
    % paste into all images (not including original)
    indexes_of_interest = find(h_all_axes &  (h_all_axes~=h_Copy_axes) )';
elseif Paste_all & Paste_new
    % paste into all images (not including original)
    indexes_of_interest = find(h_all_axes)';
elseif ~Paste_all & Paste_new    
    % paste only into target image
    indexes_of_interest = find(h_all_axes==h_Paste_axes);
else %~Paste_all & ~Paste_new
    % paste only into target image
    indexes_of_interest = find(h_all_axes==h_Paste_axes & (h_Paste_axes~=h_Copy_axes) );
end;

% Don't do anything if Paste command is bad
if ~isempty(indexes_of_interest)
    update_list = [];
    for i = indexes_of_interest,
        % for each new Profile created, copy over the Elements
        %    [h_line, h_end1, h_end2, h_center, h_number] -> 2D profiles
        %    [h_center, h_number]                         -> Temporal profiles
        for j = 1:length(i_Copy_Profile.Profile_Elements)
            h(j) = copyobj(i_Copy_Profile.Profile_Elements(j), h_all_axes(i));
        end;
        
        % if the Profile already exists, erase it before writing a new one in its place!
        if ~isempty(hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Profile_Exists)
            delete([hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Profile_Elements, ...
                    hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Single_Line_Elements, ...
                    hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Multiple_Line_Elements]);
            
            % Prepare the fields for the creation of new profiles
            hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Single_Line_Elements = [];
            hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Multiple_Line_Elements = [];
        end;
       
        hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Profile_Exists   = 1;
        hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Profile_Type     = i_Copy_Profile.Profile_Type;
        hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Profile_Color    = hT.colororder(i_Copy_Profile.Current_Profile);
        hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).N                = i_Copy_Profile.N;
        
        hT.Profile_info_table(i_Copy_Profile.Current_Profile, i).Profile_Elements = h;
        
        % make sure the objects composing the Profile have the correct indices in the Userdata 
        % for action identification
        set(h, 'UserData', [i_Copy_Profile.Current_Profile, i], ...
            'Color', hT.colororder(i_Copy_Profile.Current_Profile));
        set(h(end), 'String', num2str(i_Copy_Profile.Current_Profile));
        
        % now add the indexes of the created ROIs to the update list
        update_list(end+1,:) = [i_Copy_Profile.Current_Profile, i];     
        
    end;    
        
    guidata(findobj('Tag', 'Prof_T_figure'), hT);
    Update_Profile_Info(update_list);
    Update_Profile_Info_String(update_list);
    Resort_Profile_Info_Listbox;
    Change_Current_Profile([update_list(1,:)]);
    Highlight_Current_Profile([update_list(1,:)]);
    Update_Profile_Plot(update_list);
    figure(hT.Parent_figure);
    figure(hT.Tool_figure);
    figure(hT.Info_figure);
    figure(hT.Plot_figure);
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Delete_Profile(scope);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to delete a Profile. Check scope of deletion:
%  current Profile 
%  All Profiles in current Image
%  All Profiles with same number
%  All Profiles

%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

hT = guidata(findobj('Tag', 'Prof_T_figure'));
CURRENT_PROFILE  = 1; ALL_IMAGE_PROFILES   = 2; ALL_CURRENT_PROFILES = 3; ALL_PROFILES     = 4;

if nargin < 1  % call from edit ROI
	scope = get(hT.Delete_Profile_popupmenu, 'Value');
end;

update_list = [];
Profiles_remaining = 0;
% don't want to try to erase if there is no current profile
% unless all profiles are to be deleted
if (~isempty(hT.Current_Profile)) | (isempty(hT.Current_Profile) & (scope==ALL_PROFILES | scope == ALL_IMAGE_PROFILES) )
    new_Current_Profile = [];
    switch scope
    case CURRENT_PROFILE 
        delete([hT.Profile_info_table(hT.Current_Profile(1), hT.Current_Profile(2)).Profile_Elements, ...
                hT.Profile_info_table(hT.Current_Profile(1), hT.Current_Profile(2)).Single_Line_Elements, ...
                hT.Profile_info_table(hT.Current_Profile(1), hT.Current_Profile(2)).Multiple_Line_Elements]);

        hT.Profile_info_table(hT.Current_Profile(1), hT.Current_Profile(2)) = Init_Profile_info_table;
        update_list = hT.Current_Profile;
               
    case ALL_IMAGE_PROFILES 
        % find current index of ROI
        Delete_Axes = find(hT.h_all_axes'== hT.Current_Axes);
        
        % Make the update_list
        update_list = [];
        for i = 1:size(hT.Profile_info_table,1)
            if ~isempty(hT.Profile_info_table(i,Delete_Axes).Profile_Exists)
                update_list = [update_list; get(hT.Profile_info_table(i,Delete_Axes).Profile_Elements(1), 'Userdata')];
            end;
        end;
        
        % Remove the profile components
        delete([hT.Profile_info_table(:,Delete_Axes).Profile_Elements, ...
                hT.Profile_info_table(:,Delete_Axes).Single_Line_Elements, ...
                hT.Profile_info_table(:,Delete_Axes).Multiple_Line_Elements]);
        [hT.Profile_info_table(:,Delete_Axes)]  = deal(Init_Profile_info_table);
        
        if ~isempty(hT.Current_Profile) & (Delete_Axes ~= hT.Current_Profile(2))
            new_Current_Profile = hT.Current_Profile;
        end;
        
    case ALL_CURRENT_PROFILES
        
        Delete_Profile = hT.Current_Profile(1);

        % Make the update_list
        update_list = [];
        for i = 1:size(hT.Profile_info_table,2)
            if ~isempty(hT.Profile_info_table(Delete_Profile, i).Profile_Exists)
                update_list = [update_list; get(hT.Profile_info_table(Delete_Profile, i).Profile_Elements(1), 'Userdata')];
            end;
        end;

        delete([hT.Profile_info_table(Delete_Profile,:).Profile_Elements, ...
                hT.Profile_info_table(Delete_Profile,:).Single_Line_Elements, ...
                hT.Profile_info_table(Delete_Profile,:).Multiple_Line_Elements]);
        
        [hT.Profile_info_table(Delete_Profile, :)] = deal(Init_Profile_info_table);
        
    case ALL_PROFILES
            delete([hT.Profile_info_table(:).Profile_Elements, ...
                    hT.Profile_info_table(:).Single_Line_Elements, ...
                    hT.Profile_info_table(:).Multiple_Line_Elements]);
            [hT.Profile_info_table(:)]  = deal(Init_Profile_info_table);						
            Profiles_remaining = 0;     
            
    end;

    ht.Current_Profile = new_Current_Profile;
    
    % check if there are any Profiles left
    for i = 1:size(hT.Profile_info_table,1)
        % do not attempt to delete if the whole row of ROIs is empty (been deleted)
        if ~isempty(find([hT.Profile_info_table(i,:).Profile_Exists]))
            Profiles_remaining = 1;    
        else
            [hT.Profile_info_table(i,:).Profile_Exists] = deal([]);
        end;       
    end;
    
    if Profiles_remaining
        % if there are Profiless left
        guidata(findobj('Tag', 'Prof_T_figure'), hT);
        %Update_Profile_Info(update_list);
        Update_Profile_Info_String(update_list);
        Resort_Profile_Info_Listbox;
        Highlight_Current_Profile(new_Current_Profile);
		Toggle_Axes_View;
    else
        % if there are no Profiless left, close the Profile info figure;
        % turn buttons since there are no profiles left
        Change_Object_Enable_State(hT, 'Off',1);
        hT.Profile_info_table = Init_Profile_info_table;
        % now close the info and plot windows
        delete(hT.Info_figure);
        delete(hT.Plot_figure);
        hT.Info_figure = [];
        hT.Plot_figure = [];
        hT.handlesInfo = [];
        hT.handlesPlot = [];
        guidata(findobj('Tag', 'Prof_T_figure'), hT);
    end;    
    
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Change_Current_Axes;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update the current axes when user clicks on an axes
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;
h_axes= gca;
handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
handlesTool.Current_Axes = h_axes;
guidata(findobj('Tag', 'Prof_T_figure'), handlesTool);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Change_Current_Profile(Profile_info);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Update the current ROI when user clicks on an active object
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;

hT = guidata(findobj('Tag', 'Prof_T_figure'));
if nargin==0
	% set the current profile
    hT.Current_Profile = get(gco, 'UserData');
	guidata(findobj('Tag', 'Prof_T_figure'),hT);
	
	% Check on the selection type. If an alternate selection was used, then
	% stretch the profile to the boundary & update all info about it
	SelectionType = get( hT.Parent_figure, 'SelectionType');	
	if strcmp(SelectionType,'alt')
		% Right clicked the profile - stretches the line to the edges.
		update_list = Stretch_Current_Profile(hT);	
		Update_Profile_Info(update_list);
		Update_Profile_Info_String(update_list);
		Resort_Profile_Info_Listbox;
		Update_Profile_Plot(update_list);
		figure(hT.Parent_figure);
		figure(hT.Tool_figure);
		figure(hT.Info_figure);
		figure(hT.Plot_figure);	
	end;
else
	
	hT.Current_Profile = Profile_info;
	guidata(findobj('Tag', 'Prof_T_figure'),hT);

end;
% now select the current ROI in the information windows
Highlight_Current_Profile(hT.Current_Profile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Highlight_Current_Profile(current_profile);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function, ' Profile_info=', mat2str(current_profile)]); end;

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
current_page = get(handlesTool.handlesInfo.Profile_Info_listbox,'String'); 
if ~isempty(current_profile)
    current_string = squeeze(handlesTool.String_info_table(current_profile(1,1), current_profile(1,2),:))';
    for i = 1:size(current_page, 1)
        if strcmp(current_string(1,1:end-1), current_page(i,:))
            set(handlesTool.handlesInfo.Profile_Info_listbox, 'Value', i);
        end
    end;
else
    % want to set current ROI to blank (due to deletion of current ROI)
    set(handlesTool.handlesInfo.Profile_Info_listbox, 'Value', []);
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Profile_Size_Adjust_Entry(origin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
fig = gcf;
set(fig, 'WindowButtonMotionFcn', ['Profile_tool(''Profile_Size_Adjust'',', num2str(origin), ');']);
set(fig,'WindowButtonUpFcn', ['Profile_tool(''Profile_Size_Adjust_Exit'')']);
Change_Current_Axes;
Profile_Size_Adjust(origin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Profile_Size_Adjust(origin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function, ' | origin:', num2str(origin)]); end;
% Origin = 2 if end2 callback

point = get(gca,'CurrentPoint');
hT = guidata(findobj('Tag', 'Prof_T_figure'));

Current_Profile = get(gco, 'UserData');
Profile_Type    = hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Type;

if origin == 1  % end1 callback
    x1 = point(1,1); 
    y1 = point(1,2);
    x2 = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(3), 'xdata');
    y2 = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(3), 'ydata');
else            % end2 callback
    x2 = point(1,1); 
    y2 = point(1,2);
    x1 = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(2), 'xdata');
    y1 = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(2), 'ydata');
end;

% Profiles was created as HORIZONTAL OR VERTICAL, limit its range of motion
if     hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Type == 2  % HORIZONTAL
    if origin == 1, y1 = y2;
    else            y2 = y1; 
    end;
elseif hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Type == 3  % VERTICAL
    if origin == 1, x1 = x2;
    else            x2 = x1; 
    end;  
end;
    
%disp(['Pos: = ', num2str([x1,y1,x2,y2]), ' ; Type = ', num2str(Profile_Type)]);

[c_x, c_y, n_x, n_y] = Determine_Profile_Elements(x1, y1, x2, y2, hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).N);
hT.Current_Profile = Current_Profile;
set(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(1), 'Xdata', [x1 x2], 'Ydata', [y1 y2]);
set(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(2), 'Xdata', x1, 'Ydata', y1);
set(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(3), 'Xdata', x2, 'Ydata', y2);
set(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(4), 'Xdata', c_x, 'Ydata', c_y);
set(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(5), 'Position', [n_x, n_y, 0]);

Quick_Update_Profile_Plot(hT, Current_Profile);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Profile_Size_Adjust_Exit;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

Current_Profile = get(gco, 'Userdata');
hT = guidata(findobj('Tag', 'Prof_T_figure'));

apply_all = get(hT.Link_Profile_togglebutton, 'Value');
update_list = [];

if apply_all
    for i = 1:length(hT.h_all_axes(find(hT.h_all_axes)))       
		 if ~isempty(hT.Profile_info_table(Current_Profile(1),i).Profile_Exists) & hT.Profile_info_table(Current_Profile(1),i).Profile_Exists
			 for j = 1:4
				 set(hT.Profile_info_table(Current_Profile(1),i).Profile_Elements(j), ...
					 'Xdata', get(hT.Profile_info_table(Current_Profile(1),Current_Profile(2)).Profile_Elements(j),'Xdata') ,...
					 'Ydata', get(hT.Profile_info_table(Current_Profile(1),Current_Profile(2)).Profile_Elements(j),'Ydata'));
			 end;
			 set(hT.Profile_info_table(Current_Profile(1),i).Profile_Elements(5), ...
				 'Position', get(hT.Profile_info_table(Current_Profile(1),Current_Profile(2)).Profile_Elements(5),'Position'))
			 
			 update_list(end+1,:) = [Current_Profile(1), i];
        end;
    end;
else
	%disp(['Not all'])
	update_list = Current_Profile;
end;

Change_Current_Profile(Current_Profile);
Update_Profile_Info(update_list);
Update_Profile_Info_String(update_list);
Resort_Profile_Info_Listbox;
Highlight_Current_Profile(Current_Profile);
Update_Profile_Plot(update_list);
figure(hT.Tool_figure);
figure(hT.Info_figure);
figure(hT.Plot_figure);
figure(hT.Parent_figure);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Profile_Pos_Adjust_Entry(origin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
fig = gcf;
set(fig, 'WindowButtonMotionFcn', ['Profile_tool(''Profile_Pos_Adjust'',', num2str(origin), ');']);
set(fig,'WindowButtonUpFcn', ['Profile_tool(''Profile_Pos_Adjust_Exit'')']);
Change_Current_Axes;
Profile_Pos_Adjust(origin);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Profile_Pos_Adjust(origin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
% Origin = 1 if center callback
% Origin = 2 if number callback

point = get(gca,'CurrentPoint');
x1 = point(1,1);
y1 = point(1,2);
Current_Profile = get(gco, 'Userdata');
hT = guidata(findobj('Tag', 'Prof_T_figure'));

elems = 4;
if hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Type==4
	elems = 1;
end;

if origin == 1
    x2 = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(elems), 'xdata');
    y2 = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(elems), 'ydata');
else
    t = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(elems+1), 'Position');
    x2 = t(1);
    y2 = t(2);
end;

delta = [x1 - x2, y1 - y2];
for i=1:elems
	t = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(i), {'Xdata', 'Ydata'});
	set(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(i), 'Xdata', t{1}+delta(1), 'Ydata', t{2}+delta(2));
end;
t = get(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(elems+1), 'Position');
set(hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(elems+1), 'Position', [t(1)+delta(1), t(2)+delta(2), t(3)]);
hT.Current_Profile = Current_Profile;

Quick_Update_Profile_Plot(hT, Current_Profile);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Profile_Pos_Adjust_Exit;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

Current_Profile = get(gco, 'Userdata');
hT = guidata(findobj('Tag', 'Prof_T_figure'));

elems = 4;
if hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Type==4
	elems = 1;
end;

apply_all = get(hT.Link_Profile_togglebutton, 'Value');
update_list = [];

if apply_all
    for i = 1:length(hT.h_all_axes(find(hT.h_all_axes)))       
		 if ~isempty(hT.Profile_info_table(Current_Profile(1),i).Profile_Exists) & hT.Profile_info_table(Current_Profile(1),i).Profile_Exists
			 for j = 1:elems
				 set(hT.Profile_info_table(Current_Profile(1),i).Profile_Elements(j), ...
					 'Xdata', get(hT.Profile_info_table(Current_Profile(1),Current_Profile(2)).Profile_Elements(j),'Xdata') ,...
					 'Ydata', get(hT.Profile_info_table(Current_Profile(1),Current_Profile(2)).Profile_Elements(j),'Ydata'));
			 end;
			 set(hT.Profile_info_table(Current_Profile(1),i).Profile_Elements(elems+1), ...
				 'Position', get(hT.Profile_info_table(Current_Profile(1),Current_Profile(2)).Profile_Elements(elems+1),'Position'))
			 
			 update_list(end+1,:) = [Current_Profile(1), i];
        end;
    end;
else
	%disp(['Not all'])
	update_list = Current_Profile;
end;

Change_Current_Profile(Current_Profile);
Update_Profile_Info(update_list);
Update_Profile_Info_String(update_list);
Resort_Profile_Info_Listbox;
Highlight_Current_Profile(Current_Profile);
Update_Profile_Plot(update_list);
figure(hT.Tool_figure);
figure(hT.Info_figure);
figure(hT.Plot_figure);
figure(hT.Parent_figure);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Resort_Profile_Info_Listbox(Sort_Order);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to reassemble the String_info_table into a page 
% for display in the listbox; selects the current Profile;
% if called by the toggle button, handle is sent in, if 
% called after the addition of data to the table,
% then no handle is sent in.
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
if nargin == 0 
    if get(handlesTool.handlesInfo.Sort_Image_togglebutton, 'Value')
        Sort_Order = 'Image'; 
    else 
        Sort_Order = 'Profile';
    end;
    highlight_flag = 0;
else
    highlight_flag = 1;
end;

if strcmp('Profile', Sort_Order)
    % String info table is sorted by Profile number by default. Permute
    % if sorting by image
    String_info_table = permute(handlesTool.String_info_table, [2 1 3]);
else  %Profile or other
    String_info_table = handlesTool.String_info_table;
end;

st = size(String_info_table);
String_info_table = reshape(String_info_table, st(1)*st(2), st(3));

% now deblank empty rows
% assumes last digit in row is not space in normal strings
if (size(String_info_table, 1)>1)
    g = find(String_info_table(:,size(String_info_table,2))'=='x');
    String_info_table = String_info_table(g,:);
end;
set(handlesTool.handlesInfo.Profile_Info_listbox,'String', String_info_table(:,1:end-1));

if highlight_flag
    Highlight_Current_Profile(handlesTool.Current_Profile);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Listbox_Change_Current_Profile;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
h_listbox = gcbo;

hT = guidata(findobj('Tag', 'Prof_T_figure'));

% now determine the index of the current ROI
str = get(h_listbox, {'Value', 'String'});
values = str{1}; str = str{2};

if  length(values) >1
    % too many things are highlighted, highlight only last one
    values = values(end);
    set(h_listbox,'Value', values);
end;
% avoid problems if string is empty as all Profiles are deleted 
% and there can't be a current Profile
if ~isempty(str)
    % take first 8 characters, and convert to two numbers 
    hT.Current_Profile = fliplr(str2num(str(values,1:8)));
    hT.Current_Axes = get(hT.Profile_info_table(hT.Current_Profile(1), hT.Current_Profile(2)).Profile_Elements(1), 'Parent');    
end

guidata(findobj('Tag', 'Prof_T_figure'), hT);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Toggle_Axes_View(desired_view);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to switch between multiple or single axes views in profile
% plot figure. Input, desired_view, can either be 'Single', 'Multiple', or
% empty. In the case of empty, the view is decided by the current settings
% of the uicontrols.
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
h_togglebutton = gcbo;

if nargin <1
    % Empty desired_view
    desired_view = 'Multiple';
    if get(handlesTool.handlesPlot.Single_Axes_View_togglebutton, 'Value')
        %Single Plot Modes
        desired_view = 'Single';
    end;    
else
	if strcmp(desired_view, 'Single') & ~get(handlesTool.handlesPlot.Single_Axes_View_togglebutton, 'Value')
		desired_view = 'Multiple';
	end;
	if strcmp(desired_view, 'Multiple') & ~get(handlesTool.handlesPlot.Multiple_Axes_View_togglebutton, 'Value')
		desired_view = 'Single';
	end;
end;

if strcmp(desired_view, 'Single')
    set(handlesTool.handlesPlot.h_plot_axes(find(handlesTool.handlesPlot.h_plot_axes)), 'Visible', 'off');
    t = allchild(handlesTool.handlesPlot.h_plot_axes(find(handlesTool.handlesPlot.h_plot_axes)));
    if iscell(t), set(cell2mat(t), 'Visible', 'off');
    else          set(t,           'Visible', 'off');
    end;
    set(handlesTool.handlesPlot.Single_Profile_Plot_axes, 'Visible', 'on');
    set(allchild(handlesTool.handlesPlot.Single_Profile_Plot_axes),   'Visible', 'on');
elseif strcmp(desired_view, 'Multiple');
    set(handlesTool.handlesPlot.h_plot_axes(find(handlesTool.handlesPlot.h_plot_axes)), 'Visible', 'on');
    t = allchild(handlesTool.handlesPlot.h_plot_axes(find(handlesTool.handlesPlot.h_plot_axes)));
    if iscell(t), set(cell2mat(t), 'Visible', 'on');
    else          set(t,           'Visible', 'on');
    end;  
    set(handlesTool.handlesPlot.Single_Profile_Plot_axes, 'Visible', 'off');
    set(allchild(handlesTool.handlesPlot.Single_Profile_Plot_axes),   'Visible', 'off');
	
	hp = handlesTool.handlesPlot.h_plot_axes';
	for i = 1:size(handlesTool.Profile_info_table,2)
		
		if isempty([handlesTool.Profile_info_table(:,i).Profile_Exists]);
			set(hp(i), 'Visible', 'off');
		end;
		
	end;
	
end;  
Update_Edit_Boxes;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Toggle_FFT_View;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to switch between displaying the actual data and displaying the 
% abs(fft(fftshift())) of the data. Upon calling function, the data is
% replaced for all axes, and ylims are set to auto. 
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

hT = guidata(findobj('Tag', 'Prof_T_figure'));
update_list = [];
% need to update all plots since new data is to be displayed
for i = 1:size(hT.Profile_info_table,1)
	for j = 1:size(hT.Profile_info_table,2)
		if ~isempty(hT.Profile_info_table(i,j).Profile_Exists)
			update_list(end+1,:) = [i,j];
		end;
	end;
end;    

% call profile update with the update list
%update_list = Update_Profile_Info;
%Update_Profile_Info_String(update_list);
%Resort_Profile_Info_Listbox;
%Highlight_Current_Profile(Current_Profile);
Update_Profile_Plot(update_list);

% force update the ylims due to new scale
h_radiobutton = hT.handlesPlot.Auto_Y_radiobutton;
set(h_radiobutton, 'Value', 1);
Auto_Axes_Limits('Y');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Toggle_Profile_View(desired_view);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to switch between multiple or single axes views in profile
% plot figure. Input, desired_view, can either be 'Single', 'Multiple', or
% empty. In the case of empty, the view is decided by the current settings
% of the uicontrols.
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));

Current_Profile = handlesTool.Current_Profile;

if nargin <1
	% Empty desired_view
	desired_view = 'All';
	if get(handlesTool.handlesPlot.Active_Profiles_View_togglebutton, 'Value')
		%Single Plot Modes
		desired_view = 'Active';
	end;    
else
	if strcmp(desired_view, 'Active') & ~get(handlesTool.handlesPlot.Active_Profiles_View_togglebutton, 'Value')
		desired_view = 'All';
	end;
	if strcmp(desired_view, 'All') & ~get(handlesTool.handlesPlot.All_Profiles_View_togglebutton, 'Value')
		desired_view = 'Active';
	end;
end;

if isempty(Current_Profile)
	desired_view = 'All';
end;

all_visible = 'on';
if strcmp(desired_view, 'Active')
	all_visible = 'off';
end;

for i = 1:size(handlesTool.Profile_info_table,1)
	for j = 1:size(handlesTool.Profile_info_table,2)
% 		if get(handlesTool.HandlesPlot.Single_Axes_View_togglebutton, 'Value')
% 			Elements = handlesTool.Profile_info_table(Current_Profile(1), Current_Profile(2)).Single_Line_Elements;
% 		else
% 			Elements = handlesTool.Profile_info_table(Current_Profile(1), Current_Profile(2)).Multiple_Line_Elements;
% 		end;			
        if ~isempty(handlesTool.Profile_info_table(i,j).Profile_Exists)
			if i == Current_Profile(1) & j == Current_Profile(2)
				% set the current_profile to visible
				set([handlesTool.Profile_info_table(i,j).Single_Line_Elements,...
						handlesTool.Profile_info_table(i,j).Multiple_Line_Elements], 'visible', 'on');
			else
				% set all other profiles to either visible or invisible 
				% depending on the desired view
				set([handlesTool.Profile_info_table(i,j).Single_Line_Elements,...
						handlesTool.Profile_info_table(i,j).Multiple_Line_Elements], 'visible', all_visible);
			end;
		end;
	end;
end;
% Call Toggle_Axes_View to make sure either the single plot or the 
% multiple plot objects are hidden correctly
Toggle_Axes_View;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Auto_Axes_Limits(XorY);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to set or reset automatic axis scaling
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

hT = guidata(findobj('Tag', 'Prof_T_figure'));

h_radiobutton = gcbo;
if strcmp(get(h_radiobutton, 'Tag'), 'FFT_checkbox')
	% call from fft-checkbox, not the auto-radiobutton itselft
	h_radiobutton = hT.handlesPlot.Auto_Y_radiobutton;
end;

if strcmp(XorY, 'X')
    mode = 'Xlimmode';
    lim = 'Xlim';
else %Y
    mode = 'Ylimmode';
    lim = 'Ylim';
end;

hT = guidata(findobj('Tag', 'Prof_T_figure'));

% If the Auto button is set, then the axes is determined automatically.
% if the Auto button is cleared, then the axes is determined by the main
% (single) axis - which is determiend automatically. Any changes in the
% edit boxes clears the Auto button.
if get(h_radiobutton, 'Value')
    set([hT.handlesPlot.Single_Profile_Plot_axes; hT.handlesPlot.h_plot_axes(find(hT.handlesPlot.h_plot_axes))], mode, 'Auto');
else
    set([hT.handlesPlot.Single_Profile_Plot_axes; hT.handlesPlot.h_plot_axes(find(hT.handlesPlot.h_plot_axes))], mode, 'Manual');
end;  
Update_Edit_Boxes;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Fix_Axes_Limits;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Functiont to transfer value entered in one of the editable text boxes to
% the plot xlim or ylims. If the auto flag is on when the plot command is
% made, the axes limits will be changed automatically
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
h_edit = gcbo;
hT = guidata(findobj('Tag', 'Prof_T_figure'));

% One of the edit boxes was used.
new_val = str2num(get(h_edit, 'String'));

% Verify the validity of the input. If not valid new_val will be empty    
if isempty(new_val)
    set(h_edit, 'String', num2str(get(h_edit, 'Userdata')));
    return;
end;

error = 0;
if findstr(get(h_edit, 'Tag'), 'X')        
    h_radiobutton = hT.handlesPlot.Auto_X_radiobutton;
    if h_edit == hT.handlesPlot.Max_X_edit
        if new_val <= get(hT.handlesPlot.Min_X_edit, 'Userdata'),  
            error=1;  
        end;
    elseif h_edit == hT.handlesPlot.Min_X_edit
        if new_val >= get(hT.handlesPlot.Max_X_edit, 'Userdata'),  
            error=1;  
        end;            
    end;    
    lim = 'xlim';
    lims = [str2num(get(hT.handlesPlot.Min_X_edit, 'String')),str2num(get(hT.handlesPlot.Max_X_edit, 'String'))];

elseif findstr(get(h_edit, 'Tag'), 'Y')
    h_radiobutton = hT.handlesPlot.Auto_Y_radiobutton;
    if h_edit == hT.handlesPlot.Max_Y_edit
        if new_val <= get(hT.handlesPlot.Min_Y_edit, 'Userdata'),  
            error=1;  
        end;
    elseif h_edit == hT.handlesPlot.Min_Y_edit
        if new_val >= get(hT.handlesPlot.Max_Y_edit, 'Userdata'),  
            error=1;  
        end;            
    end;    
    lim = 'ylim';
    lims = [str2num(get(hT.handlesPlot.Min_Y_edit, 'String')),str2num(get(hT.handlesPlot.Max_Y_edit, 'String'))];
end;

if error
    set(h_edit, 'String', num2str(get(h_edit, 'Userdata')));
    return;
else
    set(h_edit, 'Userdata', new_val);
end;

% Clear the auto radiobutton
set(h_radiobutton, 'Value', 0);

% Now use the edit boxes values to set the plot limits
set(hT.handlesPlot.h_plot_axes(find(hT.handlesPlot.h_plot_axes)), lim, lims);
set(hT.handlesPlot.Single_Profile_Plot_axes, lim, lims);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Update_Edit_Boxes;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to update edit boxes - takes the axis values from the
% single_profile axes and transfer them to the editable text boxes
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
hT = guidata(findobj('Tag', 'Prof_T_figure'));

% Get the axes limits from the appropiate plot and put them into the 
% editable text boxes
if get(hT.handlesPlot.Single_Axes_View_togglebutton, 'Value')
    % Single Plot Mode
    lim = get(hT.handlesPlot.Single_Profile_Plot_axes, {'xlim', 'ylim'});
else
    lims = get(hT.handlesPlot.h_plot_axes(find(hT.handlesPlot.h_plot_axes)), {'xlim', 'ylim'});
    xlim = [lims{:,1}];
    ylim = [lims{:,2}];
    lim{1} = [min(xlim(1:2:end)), max(xlim(2:2:end))];
    lim{2} = [min(ylim(1:2:end)), max(ylim(2:2:end))] ;   
end;
set(hT.handlesPlot.Min_X_edit, 'String', num2str(lim{1}(1)), 'Userdata',lim{1}(1) );
set(hT.handlesPlot.Max_X_edit, 'String', num2str(lim{1}(2)), 'Userdata',lim{1}(2));
set(hT.handlesPlot.Min_Y_edit, 'String', num2str(lim{2}(1)), 'Userdata',lim{2}(1));
set(hT.handlesPlot.Max_Y_edit, 'String', num2str(lim{2}(2)), 'Userdata',lim{2}(2));
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Save_Profile;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
hT = guidata(findobj('Tag', 'Prof_T_figure'));


save_mat  = get(hT.Save_MAT_checkbox, 'Value');
%save_ascii = get(hT.Save_ASC_checkbox, 'Value');
 
if save_mat
    % get the filename    
    if nargin < 2
        fname = []; pname = [];
        [fname,pname] = uiputfile('*.mat', 'Save .mat file');
    end;
    
	Profile_info_table = hT.Profile_info_table;
	Profile_string_table = hT.String_info_table;
	
    for i =1:size(Profile_info_table,1)
        for j = 1:size(Profile_info_table,2)
            if Profile_info_table(i,j).Profile_Exists
                Profile_info_table(i,j).Profile_x_coordinates = ...
                    get(Profile_info_table(i,j).Profile_Elements(1), 'xdata');
                Profile_info_table(i,j).Profile_y_coordinates = ...
                    get(Profile_info_table(i,j).Profile_Elements(1), 'ydata');
				
				elems = 4;
				if Profile_info_table(i,j).Profile_Type == 4,
					elems = 1;
				end;
				P = [];
				for k = 1:elems
					P = [P, get(Profile_info_table(i,j).Profile_Elements(k), 'xdata'), ...
							get(Profile_info_table(i,j).Profile_Elements(k), 'ydata')];
				end;					
				t = get(Profile_info_table(i,j).Profile_Elements(elems+1), 'Position');
				Profile_info_table(i,j).Other_coordinates = [P, t(1), t(2)];		
                Profile_info_table(i,j).Profile_length = Profile_info_table(i,j).Profile_Info(1);                
                Profile_info_table(i,j).Profile_mean   = Profile_info_table(i,j).Profile_Info(2);
                Profile_info_table(i,j).Profile_stdev  = Profile_info_table(i,j).Profile_Info(3);
                Profile_info_table(i,j).Profile_min    = Profile_info_table(i,j).Profile_Info(4);
                Profile_info_table(i,j).Profile_max    = Profile_info_table(i,j).Profile_Info(5);
            end;
        end;
    end;
    if ~isempty(fname)
        save([pname fname],  'Profile_info_table', 'Profile_string_table'); 
    end;
end;

% UNIMPLEMENTED
% if save_ascii
% 	if nargin < 2
%         fname = []; pname = [];
%         [fname,pname] =  uiputfile('*.txt', 'Save text file'); 
%     end;
%     fid = fopen([pname, [fname, '.txt']],'w');
%     for i = 1:size(ROI_string_table,1)
%         fprintf(fid, '%s\n', ROI_string_table(i,:)) ;
%     end;
%     fclose(fid);
% end;    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Load_Profile(pathname,filename);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to load an old Profile.mat file
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

if nargin < 2
    [filename, pathname] = uigetfile('*.mat', 'Pick a Mat-file containing Profiles');
end;
    
if isequal(filename,0)|isequal(pathname,0)
	disp(['Unable to load file!']);
    return
else
    P = load([pathname, filename]);
    if ~isfield(P, 'Profile_info_table')
		disp(['Invalid Profile structure - must contain a Profile_info_table!']);
        return
    else
        % found file and it contains a Profile Table 
        % begin by restoring blank state: call delete funtion and erase all
        % existing profiles
        Delete_Profile(4);

		handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
	
		%Initialize a new set of profiles to the right size
        new_Profile_info_table = P.Profile_info_table;

        if size(new_Profile_info_table,2)>= length(find(handlesTool.h_all_axes))
            % there are more images in the new table, get rid of extras
            new_Profile_info_table = new_Profile_info_table(:,1:length(find(handlesTool.h_all_axes)));
        else
            % there are more images in current figure than in original figure,
            % extend by creating empty profiles
            new_Profile_info_table(size(new_Profile_info_table,1),length(find(handlesTool.h_all_axes))).Profile_Exists = [];    
        end;
        
		% Call function that mimics create_new_profile
        Refresh_Profiles(new_Profile_info_table);        
    end;
end;

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Close_Parent_Figure;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to make sure that if parent figure is closed, 
% the ROI info, ROI Tooland ROI Draw figures are closed too.
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

%set(findobj('Tag', 'Prof_T_figure'), 'Closerequestfcn', 'closereq');
try 
    delete(findobj('Tag','Prof_T_figure'));
end;

%set(findobj('Tag', 'Prof_I_figure'), 'Closerequestfcn', 'closereq');
try 
    delete(findobj('Tag','Prof_I_figure'));
end;
    
%set(findobj('Tag', 'Prof_P_figure'), 'Closerequestfcn', 'closereq');
try 
    delete(findobj('Tag','Prof_P_figure'));
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Menu_ROI_Tool;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Prof_Tool: ', Get_Current_Function]); end;

hNewMenu = gcbo;
checked=  umtoggle(hNewMenu);
hNewButton = get(hNewMenu, 'userdata');

if ~checked
    % turn off button
    %Deactivate Button;
    set(hNewMenu, 'Checked', 'off');
    set(hNewButton, 'State', 'off' );
else
    %Activate Button;
    set(hNewMenu, 'Checked', 'on');
    set(hNewButton, 'State', 'on' );
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%% Support Routines %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions called internally function and not as callbacks

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function h_all_axes = Find_All_Axes(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to find and sort axes in a figure - or - 
% get axes handles if array of image handles is sent in
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

if strcmp(get(varargin{1}(1), 'Type'),'figure')
    % sent in nothing; determine axes handles and sort them into the correct matrix
    h_all_axes = Sort_Axes_handles(findobj(varargin{1}(1),'Type', 'Axes'));  
else
    % sent in the image handles, already sorted into matrix; now find parent axes for each one
    % but don't include them if the image is a Colorbar
    h_images = varargin{1};
    for i =1:size(h_images,1)
        for j = 1:size(h_images,2)
            if h_images(i,j)~= 0
                if ~strcmp( get(h_images(i,j), 'Tag'), 'TMW_COLORBAR')
                    h_all_axes(i,j) = get(h_images(i,j),'Parent');
                end;
            end;
        end;
    end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function h_axes = Sort_Axes_handles(h_all_axes);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% receives a column vector of handles and 
% returns a matrix depending onthe position of 
% each image on the screen
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

% assumes axes are in a grid pattern
% so sort them by position on the figure
for i = 1:length(h_all_axes);
    position(i,:) = get(h_all_axes(i),'Position');
end;

% calculate the different number of row values and the different number of column values to 
% set the matrix size
[hist_pos_y, bins_y] = hist(position(:,1));
[hist_pos_x, bins_x] = hist(position(:,2));
hy = sum(hist_pos_y>0);
hx = sum(hist_pos_x>0) ;
[hist_pos_y, bins_y] = hist(position(:,1), hy);
[hist_pos_x, bins_x] = hist(position(:,2), hx);

%hist_pos_x = fliplr(hist_pos_x);
h_axes = zeros(hx,hy);

sorted_positions = sortrows([position, h_all_axes], [2,1]); % sort x, then y
counter = 0;
for i =1:length(hist_pos_x)
    for j = 1:hist_pos_x(i)
        sorted_positions(j+counter,6) = hx - i + 1;
    end;
    counter = counter + hist_pos_x(i);  
end;

sorted_positions = sortrows(sorted_positions,[1,2]); % sort y, then x
counter = 0;
for i =1:length(hist_pos_y)
    for j = 1:hist_pos_y(i)
        sorted_positions(j+counter,7) = i;
    end;
    counter = counter + hist_pos_y(i);
end;

for i = 1:size(sorted_positions,1)
    h_axes(round(sorted_positions(i,6)),round(sorted_positions(i,7))) = sorted_positions(i,5);
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Change_Object_Enable_State(handles, State, Paste_Flag)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

set(handles.Save_Profile_pushbutton, 'Enable', State);
set(handles.Copy_Profile_pushbutton, 'Enable', State);

set(handles.Delete_Profile_pushbutton, 'Enable', State);
set(handles.Delete_Profile_popupmenu, 'Enable', State);

set(handles.Save_MAT_checkbox, 'Enable', State);
%set(handles.Save_ASC_checkbox, 'Enable', State);

set(handles.Link_Profile_togglebutton, 'Enable', State);

if Paste_Flag
    set(handles.Paste_Profile_pushbutton, 'Enable', State);
    set(handles.Paste_All_checkbox, 'Enable', State);
    set(handles.Paste_New_checkbox, 'Enable', State);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function update_list = Update_Profile_Info(update_list)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% updates the roi info (mean, std, pixs, etc) into the ROI table
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;
handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));

if nargin == 0
    update_list = [];
    % no update list sent, so update all values in the Profile_info_table
    for i = 1:size(handlesTool.Profile_info_table,1)
        for j = 1:size(handlesTool.Profile_info_table,2)
			if ~isempty(handlesTool.Profile_info_table(i,j).Profile_Exists)
				update_list(end+1,:) = [i,j];
			end;
        end;
    end;    
end
debug_mode = 0;

for i = 1:size(update_list,1)
	if    handlesTool.Profile_info_table(update_list(i,1),update_list(i,2)).Profile_Type ~= 4
		% 2D-Profile
		% get the handles of the image, and the line circle
		h_line = handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Elements(1);
		im = get(findobj(get(h_line, 'Parent'), 'Type', 'Image'), 'CData');
		xpts = get(h_line, 'xdata');
		ypts = get(h_line, 'ydata');

		
		v1  = [ xpts(end) - xpts(1), ypts(end) - ypts(1)];
		ang = 180/pi*atan2(-v1(2), v1(1));
		
		% determine the values of the pixels "under" the line.
		% as well as their distance from the first endpoint, and
		% Profile_points interpolated curve. Use a large number to insure
		% "correct" interpolation. Subsampling will lead to underestimation
		% of maxima and overestimation of minima.
		Profile_points = max(size(im))*3;
		[Px,Py, P] = improfile(im,xpts,ypts,Profile_points,'nearest');
		dist_interp = [ 0 cumsum(sqrt( (diff(Px - Px(1)).^2 + diff(Py - Py(1)).^2)))' ];
		
		
		if debug_mode
			f = figure
			subplot(121); imagesc(im); axis image; hold on;
			title('Press Any Key To Continue');
			plot(Px,Py,'r. ');            
			subplot(122); plot(dist_interp,P,'r-'); xlabel('Distance (pixels)');ylabel('Image Intensity');
			colormap(get(handlesTool.Parent_figure, 'Colormap'));
			pause,try, close(f); end;
		end
    		
	else
		% TMEPORAL Profile
		% get the handles of the image, and the line circle
		h_center = handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Elements(1);
		%current_axes = handlesTool.Current_Axes;
		xpts = round(get(h_center, 'xdata'));
		ypts = round(get(h_center, 'ydata'));

		if  (xpts<0) | (xpts>size(handlesTool.Image_Data{update_list(i,2)},2)) | (ypts<0) | (ypts>size(handlesTool.Image_Data{update_list(i,2)},1))
			% the cursor was left outside the image area
			dist_interp = [1:size(handlesTool.Image_Data{update_list(i,2)},3)];
			P = zeros(size(dist_interp));					
			Px = xpts; Py = xpts;
		else
			% All is good - fill out the profile
			P = squeeze(handlesTool.Image_Data{update_list(i,2)}(ypts, xpts,:));						
			dist_interp = [1:length(P)];
			Px = ypts; Py = xpts;
		end;
		ang = 0;
	end
	
	mn  = mean(P);	stdev= std(double(P));
	mins = min(P);	maxs = max(P);
	len  = [max(dist_interp)];

	handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Info = ...
		[len, ang, mn, stdev, mins, maxs]; 
	handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_X  =  dist_interp;
	handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_Y  =  P';
	handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_Px =  Px;
	handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_Py =  Py;
	handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_FFT=  fft_convert(P');
end;
	
% restore the Profile_info_table with its new info
guidata(findobj('Tag', 'Prof_T_figure'),handlesTool );
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Update_Profile_Info_String(update_list)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inserts new data into the string array which is later "published"
% onto the listbox. Recevies one parameter, update_list, which consists of
% the Profile_info_table indexes of profiles to update, or is blank.
% If the update list is blank, all Profiless that exist are updated
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));

if nargin == 0
    update_list = [];
    % no update list sent, so update all values in the Profile_info_table
    for i = 1:size(handlesTool.Profile_info_table,1)
        for j = 1:size(handlesTool.Profile_info_table,2)
            update_list(end+1,:) = [i,j];
        end;
    end;    
end

for i = 1:size(update_list,1)
    Profile_Statistics = double([update_list(i,[2,1]), handlesTool.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Info]);
    % if ROI has been deleted, current info second hald will be empty will be empty; Fill with spaces
    if length(Profile_Statistics)>2
        handlesTool.String_info_table(update_list(i,1), update_list(i,2),:) = Convert_Profile_Info(Profile_Statistics);
    else
        handlesTool.String_info_table(update_list(i,1), update_list(i,2),:) =' ';
    end;
end;
guidata(findobj('Tag', 'Prof_T_figure'),handlesTool );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Update_Profile_Plot(update_list);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to draw new profiles or generally update the profile window if
% profiles have been deleted
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;
hT = guidata(findobj('Tag', 'Prof_T_figure'));
h_plot_axes = hT.handlesPlot.h_plot_axes';
h_plot_axes = h_plot_axes(:);

for i = 1:size(update_list,1)

    if isempty(hT.Profile_info_table(update_list(i,1),update_list(i,2)).Single_Line_Elements)
        hT.Profile_info_table(update_list(i,1), update_list(i,2)).Single_Line_Elements = ...
            Plot_Profile(hT.handlesPlot, hT.Profile_info_table(update_list(i,1), update_list(i,2)), hT.handlesPlot.Single_Profile_Plot_axes);
        
        hT.Profile_info_table(update_list(i,1), update_list(i,2)).Multiple_Line_Elements = ...
            Plot_Profile(hT.handlesPlot, hT.Profile_info_table(update_list(i,1), update_list(i,2)), h_plot_axes(update_list(i,2)));    
        
    else

        % Profiles exist - simply update the x & y data of the profiles
        X = hT.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_X;
		if ~get(hT.handlesPlot.FFT_checkbox, 'Value')
			Y = hT.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_Y;
		else
			Y = hT.Profile_info_table(update_list(i,1), update_list(i,2)).Profile_Data_FFT;
		end;
		
        mxY = find(max(Y)==Y); mxY = mxY(1);
        mnY = find(min(Y)==Y); mnY = mnY(1);
        
        % Update the line
        set([hT.Profile_info_table(update_list(i,1), update_list(i,2)).Single_Line_Elements(1), ...
             hT.Profile_info_table(update_list(i,1), update_list(i,2)).Multiple_Line_Elements(1)], ...
            'Xdata', X, 'YData', Y);
        set([hT.Profile_info_table(update_list(i,1), update_list(i,2)).Single_Line_Elements(2), ...
             hT.Profile_info_table(update_list(i,1), update_list(i,2)).Multiple_Line_Elements(2)], ...
            'Xdata', X(mxY), 'YData', Y(mxY));
        set([hT.Profile_info_table(update_list(i,1), update_list(i,2)).Single_Line_Elements(3), ...
            hT.Profile_info_table(update_list(i,1), update_list(i,2)).Multiple_Line_Elements(3)], ...
            'Xdata', X(mnY), 'YData', Y(mnY));
    end;
end;
guidata(findobj('Tag', 'Prof_T_figure'),hT );
Toggle_Axes_View;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Quick_Update_Profile_Plot(hT, Current_Profile);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;

if  hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Type ~= 4
	% 2D-Profile
	% get the handles of the image, and the line circle
	h_line = hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(1);
	im = get(findobj(get(h_line, 'Parent'), 'Type', 'Image'), 'CData');
	xpts = get(h_line, 'xdata');
	ypts = get(h_line, 'ydata');
	
	Profile_points = max(size(im))*3;
	[Px, Py, Y] = improfile(im,xpts,ypts,Profile_points,'nearest');
	if get(hT.handlesPlot.FFT_checkbox, 'Value')
		Y = fft_convert(Y);
	end;
	D = [ 0 cumsum(sqrt( (diff(Px - Px(1)).^2 + diff(Py - Py(1)).^2)))' ];	
else
	% TEMPORAL Profile
	% get the handles of the image, and the line circle
	h_center = hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Profile_Elements(1);
	xpts = round(get(h_center, 'xdata'));
	ypts = round(get(h_center, 'ydata'));
	
	if  (xpts<0) | (xpts>size(hT.Image_Data{Current_Profile(2)},2)) | (ypts<0) | (ypts>size(hT.Image_Data{Current_Profile(2)},1))
		return;
	end;
		
	Y = squeeze(hT.Image_Data{Current_Profile(2)}(ypts, xpts,:));						
	if get(hT.handlesPlot.FFT_checkbox, 'Value')
		Y = fft_convert(Y);
	end;
	D = [1:length(Y)];
end
mxY = find(max(Y)==Y); mxY = mxY(1);
mnY = find(min(Y)==Y); mnY = mnY(1);
    
set([hT.Profile_info_table(Current_Profile(1),Current_Profile(2)).Single_Line_Elements(1), ...
		hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Multiple_Line_Elements(1)], ...
	'Xdata', D, 'YData', Y);
set([hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Single_Line_Elements(2), ...
		hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Multiple_Line_Elements(2)], ...
	'Xdata', D(mxY), 'YData', Y(mxY));
set([hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Single_Line_Elements(3), ...
		hT.Profile_info_table(Current_Profile(1), Current_Profile(2)).Multiple_Line_Elements(3)], ...
	'Xdata', D(mnY), 'YData', Y(mnY));
drawnow;	


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Create_Profile_Info_Figure;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to create and initialize the profile info figure
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

close_str = [ 'hNewButton = findobj(''Tag'', ''figProfileTool'');' ...
        ' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
        ' set(hNewButton, ''State'', ''off'' );' ...
        ' else,  ' ...
        ' Profile_tool(''Deactivate_Profile_Tool'' ,hNewButton);'...
        ' set(hNewButton, ''Value'', 0);',...
        ' end;' ];

fig_info = openfig('Profile_info_figure');
set(fig_info, 'Name', 'Profile Statistics' ,...
    'Tag', 'Prof_I_figure' ,...
    'Resize' , 'On',...
    'CloseRequestfcn', close_str...
    );

handlesInfo = guihandles(fig_info);
call_str = get(handlesInfo.Sort_Image_togglebutton, 'Callback');
set(handlesInfo.Sort_Image_togglebutton,   'Userdata', handlesInfo.Sort_Profile_togglebutton, ...
    'Callback', ['set(get(gcbo, ''UserData''), ''Value'', ~get(gcbo, ''Value''));', call_str]);
call_str = get(handlesInfo.Sort_Profile_togglebutton, 'Callback');
set(handlesInfo.Sort_Profile_togglebutton, 'Userdata', handlesInfo.Sort_Image_togglebutton, ...
    'Callback', ['set(get(gcbo, ''UserData''), ''Value'', ~get(gcbo, ''Value''));', call_str]);

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
handlesTool.handlesInfo = handlesInfo;
handlesTool.Info_figure = fig_info;

for i = 1:size(handlesTool.Profile_info_table,1)
    for j = 1:size(handlesTool.Profile_info_table,2)
        if handlesTool.Profile_info_table(i,j).Profile_Exists
            Profile_Statistics = double(handlesTool.Profile_info_table(i,j).Profile_Info);
            % note that MATLAB automatically pads the strings with empty spaces
            String_info_table(i,j,:)= Convert_Profile_Info([j,i,Profile_Statistics]);
        end;
    end
end;
% now store the String table
handlesTool.String_info_table = String_info_table;
guidata(findobj('Tag', 'Prof_T_figure'), handlesTool);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Create_Profile_Plot_Figure;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to create and initialize the Profile Plotfigure
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

close_str = [ 'hNewButton = findobj(''Tag'', ''figProfileTool'');' ...
        ' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
        ' set(hNewButton, ''State'', ''off'' );' ...
        ' else,  ' ...
        ' Profile_tool(''Deactivate_Profile_Tool'' ,hNewButton);'...
        ' set(hNewButton, ''Value'', 0);',...
        ' end;' ];

fig_plot = openfig('Profile_plot_figure');
set(fig_plot, 'Name', 'Profiles' ,...
    'Tag', 'Prof_P_figure' ,...
    'Resize' , 'On',...
    'CloseRequestfcn', close_str...
    );

handlesPlot = guihandles(fig_plot);

set(fig_plot, 'Renderer', 'zbuffer');
set(handlesPlot.Single_Profile_Plot_axes, 'Drawmode', 'Fast', 'Userdata', {'auto', 'auto'});

call_str = get(handlesPlot.Single_Axes_View_togglebutton, 'Callback');
set(handlesPlot.Single_Axes_View_togglebutton,   'Userdata', handlesPlot.Multiple_Axes_View_togglebutton, ...
    'Callback', ['set(get(gcbo, ''UserData''), ''Value'', ~get(gcbo, ''Value''));', call_str]);
call_str = get(handlesPlot.Multiple_Axes_View_togglebutton, 'Callback');
set(handlesPlot.Multiple_Axes_View_togglebutton, 'Userdata', handlesPlot.Single_Axes_View_togglebutton, ...
    'Callback', ['set(get(gcbo, ''UserData''), ''Value'', ~get(gcbo, ''Value''));', call_str]);

call_str = get(handlesPlot.All_Profiles_View_togglebutton, 'Callback');
set(handlesPlot.All_Profiles_View_togglebutton,   'Userdata', handlesPlot.Active_Profiles_View_togglebutton, ...
    'Callback', ['set(get(gcbo, ''UserData''), ''Value'', ~get(gcbo, ''Value''));', call_str]);
call_str = get(handlesPlot.Active_Profiles_View_togglebutton, 'Callback');
set(handlesPlot.Active_Profiles_View_togglebutton, 'Userdata', handlesPlot.All_Profiles_View_togglebutton, ...
    'Callback', ['set(get(gcbo, ''UserData''), ''Value'', ~get(gcbo, ''Value''));', call_str]);

handlesTool = guidata(findobj('Tag', 'Prof_T_figure'));
handlesTool.handlesPlot = handlesPlot;
handlesTool.Plot_figure = fig_plot;
handlesTool = Create_Multiple_Axes(handlesTool);
multiple_plot_axes =handlesTool.handlesPlot.h_plot_axes';

for i = 1:size(handlesTool.Profile_info_table,1)
    for j = 1:size(handlesTool.Profile_info_table,2)
        if handlesTool.Profile_info_table(i,j).Profile_Exists
            % Plot the data on the axis;
			handlesTool.Profile_info_table(i,j).Single_Line_Elements = ...
				Plot_Profile(handlesPlot, handlesTool.Profile_info_table(i,j), handlesPlot.Single_Profile_Plot_axes);
			handlesTool.Profile_info_table(i,j).Multiple_Line_Elements = ...
				Plot_Profile(handlesPlot, handlesTool.Profile_info_table(i,j), multiple_plot_axes(j) );      
		end;
    end
end;

guidata(findobj('Tag', 'Prof_T_figure'), handlesTool);
Toggle_Axes_View('Single');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function hT = Create_Multiple_Axes(hT);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Funtion used to create a series of axes underneath the main axes in the
% plot window. 
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

% Get the current plot position
pos = get(hT.handlesPlot.Single_Profile_Plot_axes, 'Position');
grid_size = size(hT.h_all_axes);

% assume size_x by size_y grid size and determine the 
% individual axes width and height. Shrink the axes to allow numbers to fit
spacex = 0.25;     spacey = 0.25;

total_w = grid_size(2) + (grid_size(2)-1)*spacex;
total_h = grid_size(1) + (grid_size(1)-1)*spacex;

w = pos(3) / total_w;
h = pos(4) / total_h;

wstep = w * spacex + w;
hstep = h * spacey + h;

h_plot_axes = zeros(size(hT.h_all_axes));
% Draw horizontal, then vertical axes
for j = 1:grid_size(1)
    for i = 1:grid_size(2)
        if hT.h_all_axes(j,i)
            h_plot_axes(j,i) = copyobj(hT.handlesPlot.Single_Profile_Plot_axes, hT.Plot_figure);
            set(h_plot_axes(j,i), ...
                'Position', [pos(1)+(i-1)*(wstep), (pos(2)+pos(4)-h -(j-1)*hstep)  , w, h]);
        end;                
    end
end
% Now have the handles to all secondary axes
hT.handlesPlot.h_plot_axes = h_plot_axes;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Info_string = Convert_Profile_Info(Info_numbers)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function that converts the information from the ROI info table (numbers)
% into a string that can be ins14erted in a cell array for display in the
% list box.
% temp fixed spacings
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;
a = [     sprintf('%2d', Info_numbers(1))];
a = [a,   sprintf('%5d', Info_numbers(2))];

total_use_digits = 8;
if Info_numbers(6) < 10^5 
    after_decimal_precision_digits = 2;
else
    after_decimal_precision_digits = 0;
end;
total_precision_digits =  total_use_digits - after_decimal_precision_digits;

a = [a,  FixLengthFormat(Info_numbers(3),total_use_digits+2, after_decimal_precision_digits)];
a = [a,  FixLengthFormat(Info_numbers(4),total_use_digits+1 , after_decimal_precision_digits)];

a = [a,  FixLengthFormat(Info_numbers(5),total_use_digits   , after_decimal_precision_digits)];

a = [a,  FixLengthFormat(Info_numbers(6),total_use_digits, after_decimal_precision_digits)];
    %sprintf('%6s',  num2str(Info_numbers(5), total_precision_digits) )];

a = [a,  FixLengthFormat(Info_numbers(7),total_use_digits-2, after_decimal_precision_digits)];
a = [a,  FixLengthFormat(Info_numbers(8),total_use_digits, after_decimal_precision_digits)];
Info_string = [a, 'x'];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function value = FixLengthFormat(num,totalChars, precision)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   By Patrick 'fishing boy' Helm
%	FixLengthFormat(num,totalChars, precision)
%		num is the number to format.	
%		totalChars is number of characters in conversion.
%				The number if right justified if the number
%				of characters is greater than the length of
%				num2str(num)
%		precision is number of digits after decimal
%
tNum = num2str(num, 16);
% find the decimal point index, if it exists
iDecimal = find(tNum == '.');
if (isempty(iDecimal)) 
    % set decimal to end of number if none 
    iDecimal = length(tNum);
    precision = 0;
else
    % add on zeroes until precision requirement is met
    while((length(tNum) - iDecimal) < 16)
        tNum = [tNum,'0'];
    end
end
% insure that even if function fails, blanks are returned
value = blanks(totalChars);
% copy character version onto output, 
% maintaining right justification
if ((iDecimal + precision) <= totalChars)
   startPos = totalChars - (precision + iDecimal) + 1;
   value(startPos:totalChars) = tNum(1:(iDecimal+precision));
end
    


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Refresh_Profiles(new_Profile_info_table);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function creates new Profiles after they have been loaded from mat file.
% It closely mimics the create_new_profile function and assumes a clean
% slate as delete_profiles(all) was called in load_profile function
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;

% load data of interest
fig_tool = findobj('Tag', 'Prof_T_figure');
handlesTool = guidata(fig_tool);

GENERAL = 1; HORIZONTAL = 2; VERTICAL = 3; TEMPORAL = 4;

first_Profile_flag = 0;
% if all Profile exist fields are empty, then this is the first Profile
if isempty([handlesTool.Profile_info_table(:).Profile_Exists])
	% create new Profile_info_table to correct size (rows = number of Profiles, columns = images) 
	% now make the invalid last in case grid is not complete
	handlesTool.Profile_info_table = repmat(Init_Profile_info_table, [1, length(find(handlesTool.h_all_axes))]);
    first_Profile_flag = 1;       
end;
Current_Profile_index = Determine_New_Index(handlesTool.Profile_info_table);

update_list = [];
h_all_axes = handlesTool.h_all_axes';
% For every element of the new table
for i = 1:size(new_Profile_info_table,1)
	for j = 1:size(new_Profile_info_table,2)
		% if the Profile_exists
		if ~isempty(new_Profile_info_table(i,j).Profile_Exists)
			switch new_Profile_info_table(i,j).Profile_Type
				case {HORIZONTAL, VERTICAL, GENERAL}
					% create a general 2D profile

					set(handlesTool.Parent_figure, 'CurrentAxes', h_all_axes(j));
					set(0, 'CurrentFigure', handlesTool.Parent_figure);
					
					pos_x1   = new_Profile_info_table(i,j).Other_coordinates(5);
					pos_y1   = new_Profile_info_table(i,j).Other_coordinates(6);
					pos_x2   = new_Profile_info_table(i,j).Other_coordinates(7);
					pos_y2   = new_Profile_info_table(i,j).Other_coordinates(8);
					center_x = new_Profile_info_table(i,j).Other_coordinates(9);
					center_y = new_Profile_info_table(i,j).Other_coordinates(10);
					number_x = new_Profile_info_table(i,j).Other_coordinates(11);
					number_y = new_Profile_info_table(i,j).Other_coordinates(12);
										
					% handles_values = [ h_line, h_end1, h_end2, h_center, h_number]		
					handle_values = Make_2D_Profile_Elements(...
						center_x, center_y, ...
						pos_x1, pos_y1, ...
						pos_x2, pos_y2, ...
						number_x, number_y, ...
						handlesTool.colororder(i), ...
						i ...
					);
					
					handlesTool.Profile_info_table(i,j).Profile_Exists   = 1;
					handlesTool.Profile_info_table(i,j).Profile_Elements = handle_values;
					handlesTool.Profile_info_table(i,j).Profile_Type     = new_Profile_info_table(i,j).Profile_Type;
					handlesTool.Profile_info_table(i,j).Profile_Color    = handlesTool.colororder(i);
					handlesTool.Profile_info_table(i,j).N                = new_Profile_info_table(i,j).N;
					handlesTool.Current_Profile = [i,j];
					
					set(handle_values, 'Userdata', handlesTool.Current_Profile);
					
					update_list(end+1,:) = [i,j];
					guidata(findobj('Tag', 'Prof_T_figure'), handlesTool);
					
					drawnow
					
					
				case TEMPORAL
					% Define a temporal profile by placing a single '+' in the middle
					% of the image. If there is not temporal data in the appdata, then
					% simply display the point value
										
					set(handlesTool.Parent_figure, 'CurrentAxes', h_all_axes(j));
					set(0, 'CurrentFigure', handlesTool.Parent_figure);

					center_x = new_Profile_info_table(i,j).Other_coordinates(1);
					center_y = new_Profile_info_table(i,j).Other_coordinates(2);
					number_x = new_Profile_info_table(i,j).Other_coordinates(3);
					number_y = new_Profile_info_table(i,j).Other_coordinates(4);
					
					h_center = plot(center_x, center_y , ...
						['+', handlesTool.colororder(i)], ...
						'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'', 1)');
					h_number = text(number_x, number_y, num2str(Current_Profile_index), ...
						'Color', handlesTool.colororder(i) , ...
						'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'', 2)');            
					
					handlesTool.Profile_info_table(i,j).Profile_Exists   = 1;
					handlesTool.Profile_info_table(i,j).Profile_Elements = [h_center, h_number];
					handlesTool.Profile_info_table(i,j).Profile_Type     = new_Profile_info_table(i,j).Profile_Type;
					handlesTool.Profile_info_table(i,j).Profile_Color    = handlesTool.colororder(i);
					handlesTool.Profile_info_table(i,j).N                = new_Profile_info_table(i,j).N;
					handlesTool.Current_Profile = [i,j];
						
					handlesTool.Current_Profile = [i,j];				
					set([h_center, h_number], 'Userdata', handlesTool.Current_Profile);
					
					update_list(end+1,:) = [i,j];
					guidata(findobj('Tag', 'Prof_T_figure'), handlesTool);
					
					drawnow
					
			end;
		end;
	end;
end;
			
% call the Profile info update function: puts data into Profile_info_table
Update_Profile_Info(update_list);

if first_Profile_flag
	% creates figure the first time and creates the string table that is to be
	% used for "publishing" the ROI data
	Create_Profile_Info_Figure;
	Create_Profile_Plot_Figure;
	Change_Object_Enable_State(handlesTool,'Off',1);
	Change_Object_Enable_State(handlesTool,'On',0);
	% Don't need to update string if this is the first ROI
else
	% call function that will take info string table and "publish" it			
	Update_Profile_Info_String(update_list);
end;
% update current ROI index
Resort_Profile_Info_Listbox;
Highlight_Current_Profile(handlesTool.Current_Profile);
Update_Profile_Plot(update_list);
Update_Edit_Boxes;       

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Close_Old_Figure(Figure_Name, Figure_Handle);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to close (delete if necessary) old figures. If figure 'Tag' is
% specified, then it is used. If not, the handle is used.
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

if ~isempty(Figure_Name)
	Figure_Handle = findobj('Tag', Figure_Name);
end;
if ~isempty(Figure_Handle)
	set(Figure_Handle,'CloseRequestFcn', 'closereq');
	try close(Figure_Handle);
	catch delete(Figure_Handle);
	end;
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [h_objects, object_enable_states] = Change_Figure_States(State, figure_handles,  h_objects, object_enable_states)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to set the enable states of the uicontrol objects of various
% figures. Figure handles sent in are used to find all the objects, 
% if the object handles are not sent in initially. 
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;

if nargin <3
	h_objects = findobj(figure_handles, 'type', 'uicontrol');
	object_enable_states = {};
else
	if length(h_objects) ~= length(object_enable_states)
		disp(['Can''t Enable Objects!']);;
	end;
end;

if strcmp(State, 'Disable')
	object_enable_states = get(h_objects, 'Enable');
	set(h_objects, 'Enable', 'off');
elseif strcmp(State, 'Enable')
	if isempty(object_enable_states)
		set(h_objects, 'Enable', 'On');
	else
		for i=1:length(h_objects)
			set(h_objects(i), 'Enable', object_enable_states{i});
		end;
	end;
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Profile_info_table = Init_Profile_info_table;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Function to create an empty Profile_info_table with the 
%  correct fields initialized to empty
%global DB; if DB disp(['Profile_Tool: ', Get_Current_Function]); end;

Profile_info_table = struct( ...
	'Profile_Data_X',  [], ...
	'Profile_Data_Y',  [], ...
	'Profile_Data_Px', [], ...
	'Profile_Data_Py', [], ...
	'Profile_Data_FFT', [], ...
	'Profile_Exists',  [], ...
    'Profile_Elements',[], ...
    'Profile_Type',    [], ...
	'Profile_Info',    [], ...
    'Profile_Color',   [], ...
    'Single_Line_Elements',     [], ...
    'Multiple_Line_Elements',   [], ...
    'N', []);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function  handle_values = Make_2D_Profile_Elements(center_x, center_y, ...
				pos_x1, pos_y1, ...
				pos_x2, pos_y2, ...
				number_x, number_y, ...
				profile_color, ...
				profile_number ...
				);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Funciton to create sub-elements of a Profile, including the line, the
%  two resize circles, the center marker and the Profile number marker
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;
% handle_values = [h_line, h_end1, h_end2, h_center, h_number];

h_line = plot([pos_x1, pos_x2], [ pos_y1, pos_y2], [profile_color,'-'], ...
	'ButtonDownFcn', 'Profile_tool(''Change_Current_Profile'')');

h_center = plot(center_x, center_y , [profile_color,'+'], ...
	'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'',1)'); 

h_end1 = plot(pos_x1 , pos_y1, [profile_color,'o'],...
	'ButtonDownFcn', 'Profile_tool(''Profile_Size_Adjust_Entry'',1)');
h_end2 = plot(pos_x2 , pos_y2, [profile_color,'s'],...
	'ButtonDownFcn', 'Profile_tool(''Profile_Size_Adjust_Entry'',2)');

h_number = text(number_x, number_y, num2str(profile_number),...
	'color', profile_color, ...
	'HorizontalAlignment', 'center' , ...
	'ButtonDownFcn', 'Profile_tool(''Profile_Pos_Adjust_Entry'',2)'); 

handle_values = [h_line, h_end1, h_end2, h_center, h_number];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function elements = Plot_Profile(hP, table_entry, axes);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Function to plot the profile in the given axes

X = table_entry.Profile_Data_X;
Y = table_entry.Profile_Data_Y;

if get(hP.FFT_checkbox, 'Value')
	Y = table_entry.Profile_Data_FFT;
end;

set(0, 'CurrentFigure', hP.Prof_P_figure);
set(hP.Prof_P_figure, 'CurrentAxes', axes);

elements(1) = plot(X,Y, [table_entry.Profile_Color, '-']);
%hold on;
mxY = find(max(Y)==Y); mxY = mxY(1);
mnY = find(min(Y)==Y); mnY = mnY(1);

elements(2) = plot(X(mxY), Y(mxY), [table_entry.Profile_Color, '^']);
elements(3) = plot(X(mnY), Y(mnY), [table_entry.Profile_Color, 'v']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function update_list = Stretch_Current_Profile(hT);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Function to stretch current profile upon right (alternate) clicking
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;

h_line = hT.Profile_info_table(hT.Current_Profile(1), hT.Current_Profile(2)).Profile_Elements(1);
data = get(h_line, {'xdata', 'ydata'});
xpts = data{1}; ypts = data{2};
data = get(get(h_line, 'Parent'), {'xlim', 'ylim'});
% Avoid NaNs by making right & top edges round down
xlim = data{1} + [ 0, -0.01];  ; ylim = data{2} + [ 0 , -0.01];
apply_all = get(hT.Link_Profile_togglebutton, 'Value');


if ~diff(xpts)
	m = Inf;
else
	m = diff(ypts)/ diff(xpts);
end;
b = ypts(1) - m*xpts(1);

if isinf(abs(m))
	% Vertical profile - infinite slope
	if (diff(ypts) < 0) 
		x1 = xpts(2); x2 = xpts(1);
		y1 = ylim(2); y2 = ylim(1);
	else
		x1 = xpts(1); x2 = xpts(2);
		y1 = ylim(1); y2 = ylim(2);
	end;

elseif ~m
	% horizontal profile - 
	if (diff(xpts) < 0) 
		x1 = xlim(2); x2 = xlim(1);
		y1 = ypts(2); y2 = ypts(1);
	else
		x1 = xlim(1); x2 = xlim(2);
		y1 = ypts(1); y2 = ypts(2);
	end;	
else
	% Calculate the four new possible endpoints 
	new_ypt_1a =  xlim(1)*m + b;    new_xpt_1a = xlim(1);
	new_xpt_1b = (ylim(1)   - b)/m; new_ypt_1b = ylim(1);
	
	new_ypt_2a =  xlim(2)*m + b;    new_xpt_2a = xlim(2);
	new_xpt_2b = (ylim(2)   - b)/m; new_ypt_2b = ylim(2);
	
	% if the new limit is between the axes limits then it is going to be
	% the new point. After any selection (except corner), only two ylims, 
	% one ylim and one xlim , or two xlims, should be valid
	x = []; y = []; 
	if	(new_xpt_1b >= xlim(1)) & (new_xpt_1b <= xlim(2))  
		% use the b case
		y = [y, new_ypt_1b];
		x = [x, new_xpt_1b];
	end;
	if (new_ypt_1a >= ylim(1)) & (new_ypt_1a <= ylim(2))  
		% Use the 'a' case
		y = [y, new_ypt_1a];
		x = [x, new_xpt_1a];
	end;
	if	(new_xpt_2b >= xlim(1)) & (new_xpt_2b <= xlim(2))  
		% use the b case
		y = [y, new_ypt_2b];
		x = [x, new_xpt_2b];
	end;
	if (new_ypt_2a >= ylim(1)) & (new_ypt_2a <= ylim(2))  
		% Use the 'a' case
		y = [y, new_ypt_2a];
		x = [x, new_xpt_2a];
	end;
	
	x1 = x(1); y1 = y(1);
	x2 = x(2); y2 = y(2);

	%Calculale new unit vector for direction and compare it to the old unit
	%vector; if dot product is negative, then they are in oposite
	%directions and flip the new vector.
	v1  = [ x2 - x1, y2 - y1];
	v1 = v1 / sqrt(sum(v1.^2));
	
	old_data = get(hT.Profile_info_table(hT.Current_Profile(1), hT.Current_Profile(2)).Profile_Elements(1), {'xdata' , 'ydata'});
	old_v1 = [diff(old_data{1}), diff(old_data{2})];
	old_v1 = old_v1 / sqrt(sum(old_v1.^2));
	
	if dot(v1, old_v1)<0
		x1 = x(2); y1 = y(2);
		x2 = x(1); y2 = y(1);
	end
	
end;

[c_x, c_y, n_x, n_y] = Determine_Profile_Elements(x1, y1, x2, y2, ...
	hT.Profile_info_table(hT.Current_Profile(1),hT.Current_Profile(2)).N);

% Generate the udpate list for all other functions
update_list = [];
if apply_all
	jrange = 1:size(hT.Profile_info_table,2);
else
	jrange = hT.Current_Profile(2);
end;

for j = jrange
	if hT.Profile_info_table(hT.Current_Profile(1),j).Profile_Exists
		update_list = [update_list; hT.Current_Profile(1),j];
		% Update the current Profiles
		set(hT.Profile_info_table(update_list(end,1), update_list(end,2)).Profile_Elements(1),...
			'xdata', [x1 x2], 'ydata', [y1 y2]);			
		set(hT.Profile_info_table(update_list(end,1), update_list(end,2)).Profile_Elements(2), ...
			'xdata', x1, 'ydata', y1);
		set(hT.Profile_info_table(update_list(end,1), update_list(end,2)).Profile_Elements(3), ...
			'xdata', x2, 'ydata', y2);
		set(hT.Profile_info_table(update_list(end,1), update_list(end,2)).Profile_Elements(4), ...
			'xdata', c_x, 'ydata', c_y);
		set(hT.Profile_info_table(update_list(end,1), update_list(end,2)).Profile_Elements(5), ...
			'Position', [n_x, n_y, 0]);
	end;
end;
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [index, table] = Determine_New_Index(table);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% find empty slots in ROI_info Table
index = [];
for i = size(table,1):-1:1
    % found empty slot, use it; search in reverse to use the nearest one    
    if isempty([table(i,:).Profile_Exists])
        index = i;
	end;
end;
% Didn't find an empty slot, create a new row in table
if isempty(index)
    index = size(table,1) + 1;
end;

% make sure we don't over clutter the screen
if index > 10
	msgbox('Too Many Profile Objects. Please delete some before creating new ones.');
    index = -1;
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function [c_x, c_y, n_x, n_y] = Determine_Profile_Elements(x1, y1, x2, y2, N);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Function to determine the position of 2D profile center and number
%  elements
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;
c_x = mean([x1, x2]);
c_y = mean([y1, y2]);

v1  = [ x2 - x1, y2 - y1];
v1 = v1 / sqrt(sum(v1.^2));
v1 = v1 * N;

n_x= x2 + v1(1);
n_y= y2 + v1(2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function  x = fft_convert(x);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to apply an fft to the current data
%global DB; if DB disp(['Profile_tool: ', Get_Current_Function]); end;
x = abs(fftshift(fft(double(x))));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function  func_name = Get_Current_Function;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Debug function - returns current function name
x = dbstack;
x = x(2).name;
func_name = x(findstr('(', x)+1:findstr(')', x)-1);

% func_name = [];
% for i = length(x):-1:2
% 	if ~isempty(findstr('(', x(i).name))
% 		func_name = [func_name, x(i).name(findstr('(', x(i).name)+1:findstr(')', x(i).name)-1), ' : '];
% 	end;
% end;
% func_name = func_name(1:end-3);