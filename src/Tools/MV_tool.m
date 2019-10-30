function MV_tool(varargin);
% function MV_tool(varargin);
% Movie viewing tool for displaying 3D or 4D sets of data. Use with
% imagescn or imagescn. Can export to avi.
%
% Usage: MV_tool;
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
else
   Action = varargin{1};  
end

%%global DB; DB = 1;

switch Action
case 'New'
    Create_New_Button;

case 'Activate_View_Images'
    Activate_View_Images;
     
case 'Deactivate_View_Images'
    Deactivate_View_Images(varargin{2:end});

case 'Set_Current_Axes'
	Set_Current_Axes(varargin{2:end});
	
case 'Limit'
	Limit(varargin{2:end});
		
case 'Step'
	Step(varargin{2:end});
	
case 'Set_Frame'
	Set_Frame;
	
case 'Set_Frame_Limit'
	Set_Frame_Limit;

case 'Reset_Frame_Limit'
	Reset_Frame_Limit;
	
case 'Play_Movie'
	Play_Movie;
	
case 'Stop_Movie'
	Stop_Movie;

case 'Make_Movie'
	Make_Movie;
	
case 'Show_Frames'
	Show_Frames;
	
case 'Menu_View_Images'
    Menu_View_Images;
    
case 'Close_Parent_Figure'
    Close_Parent_Figure;    
    
otherwise
    disp(['Unimplemented Functionality: ', Action]);
   
end;
      
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Create_New_Button
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

fig = gcf;

% Find handle for current image toolbar and menubar
hToolbar = findall(fig, 'type', 'uitoolbar', 'Tag','FigureToolBar' );
hToolMenu = findall(fig, 'Label', '&Tools');

if ~isempty(hToolbar) & isempty(findobj(hToolbar, 'Tag', 'figViewImages'))
	hToolbar_Children = get(hToolbar, 'Children');
	
	% The default button size is 15 x 16 x 3. Create Button Image
	button_size_x= 16;
	button_image = NaN* zeros(15,button_size_x);
	
	f= [...
			9    10    11    12    13    14    15    24    30    39    45    50    51    52    53    54    60, ...
			65    69    75    80    84    90    91    92    93    94    95    99   100   101   102   103   104, ...
			105   106   110   116   121   125   131   135   136   140   141   142   143   144   145   146   149, ...
			151   157   163   166   172   177   181   182   183   184   185   186   187   189   191   204   205, ...
			219   220   221 ...
		]; 
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
		'OnCallback', 'MV_tool(''Activate_View_Images'')',...
		'OffCallback', 'MV_tool(''Deactivate_View_Images'')',...
		'Tag', 'figViewImages', ...
		'TooltipString', 'View Images & Make Movies',...
		'UserData', [], ...
		'Separator', separator, ...
		'Enable', 'on');   
end;

% If the menubar exists, create menu item
if ~isempty(hToolMenu) & isempty(findobj(hToolMenu, 'Tag', 'menuViewImages'))
	hWindowLevelMenu = findobj(hToolMenu, 'Tag', 'menuWindowLevel');
	hPanZoomMenu     = findobj(hToolMenu, 'Tag', 'menuPanZoom');
	hROIToolMenu     = findobj(hToolMenu, 'Tag', 'menuROITool');
	hViewImageMenu   = findobj(hToolMenu, 'Tag', 'menuViewImages');
	hPointToolMenu   = findobj(hToolMenu, 'Tag', 'menuPointTool');
	hRotateToolMenu  = findobj(hToolMenu, 'Tag', 'menuRotateTool');
    hProfileToolMenu = findobj(hToolMenu, 'Tag', 'menuProfileTool');
	
	position = 9;
	separator = 'On';
	hMenus = [ hWindowLevelMenu, hPanZoomMenu,hROIToolMenu, hPointToolMenu, hRotateToolMenu, hProfileToolMenu];
	if length(hMenus>0) 
		position = position + length(hMenus);
		separator = 'Off';
	end;
	
	hNewMenu = uimenu(hToolMenu,'Position', position);
	set(hNewMenu, 'Tag', 'menuViewImages','Label',...
		'Movie Tool',...
		'CallBack', 'MV_tool(''Menu_View_Images'')',...
		'Separator', separator,...
		'UserData', hNewButton...
		); 
	h_axis = findobj(gcf, 'Type', 'Axes');
	
	if isempty(getappdata(h_axis(1), 'CurrentImage'))
		% current images do not have hidden dimension data
		set(hNewButton, 'Enable', 'off');
	end;
	
end;
	

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Activate_View_Images(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

fig = gcf;

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');
    hNewButton = gcbo;
    set(findobj('Tag', 'menuViewImages'),'checked', 'on');
else
    hNewButton = varargin{1};
end;

% allows for calls from buttons other than those in toolbar
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure'),
    fig = get(fig, 'Parent');
end

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
	PZ = findobj(hToolbar_Children,'Tag', 'figPanZoom');
	RT = findobj(hToolbar_Children,'Tag', 'figROITool');
	MV = findobj(hToolbar_Children,'Tag', 'figViewImages');
	PM = findobj(hToolbar_Children,'Tag', 'figPointTool');
	RotT = findobj(hToolbar_Children,'Tag', 'figRotateTool');
	Prof = findobj(hToolbar_Children, 'Tag', 'figProfileTool');
	
	old_ToolHandles  =     [Rot3D, ZoomO, ZoomI,WL,PZ,RT,PM,RotT,Prof];
	old_ToolEnables  = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,PM,RotT,Prof], 'Enable');
	old_ToolStates   = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,PM,RotT,Prof], 'State');
	
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

% Start PZ GUI
fig2_old = findobj('Tag', 'MV_figure');
% close the old WL figure to avoid conflicts
if ~isempty(fig2_old) close(fig2_old);end;

% open new figure
fig2_file = 'MV_tool_figure.fig';
fig2 = openfig(fig2_file,'reuse');
optional_uicontrols = { ...
    'Apply_radiobutton',    'Value'; ...
    'Frame_Rate_edit',      'String'; ...
    'Make_Avi_checkbox',    'Value'; ...
    'Make_Mat_checkbox',    'Value'; ...
    'Show_Frames_checkbox', 'Value'; ...
                   };
set(SP,'Userdata',{fig2, fig2_file, optional_uicontrols});

% Generate a structure of handles to pass to callbacks, and store it. 
handlesMV = guihandles(fig2);

close_str = [ 'hNewButton = findobj(''Tag'', ''figViewImages'');' ...
        ' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
        ' set(hNewButton, ''State'', ''off'' );' ...
        ' else,  ' ...
        ' MV_tool(''Deactivate_View_Images'',hNewButton);'...
        ' set(hNewButton, ''Value'', 0);',...
        ' end;' ];

set(fig2, 'Name', 'Image Viewing Tool',...
    'closerequestfcn', close_str);

% Record and store previous WBDF etc to restore state after PZ is done. 
old_WBDF = get(fig, 'WindowButtonDownFcn');
old_WBMF = get(fig, 'WindowButtonMotionFcn');
old_WBUF = get(fig, 'WindowButtonUpFcn');
old_UserData = get(fig, 'UserData');
old_CRF = get(fig, 'Closerequestfcn');

% Store initial state of all axes in current figure for reset
h_all_axes = flipud(findobj(fig,'Type','Axes'));
h_axes = h_all_axes(1);

set(h_all_axes, 'ButtonDownFcn', 'MV_tool(''Set_Current_Axes'')');
set(0,'currentfigure', fig);
%set(fig, 'CurrentAxes', h_axes);


handlesMV.Axes = h_all_axes;
visibility = get(handlesMV.Show_Frames_checkbox, 'Value');
if visibility, visibility = 'On' ;
else           visibility = 'Off'; end;
textFontSize = 20;
figUnits = get(fig,'Units');
set(fig, 'Units', 'inches');
figSize = get(fig, 'position');
reffigSize = 8;   % 8 inch figure gets 20 pt font
textFontSize = textFontSize * figSize(3) / reffigSize;
set(fig, 'Units', figUnits);

for i = 1:length(h_all_axes)
	set(findobj(h_all_axes(i), 'Type', 'image'), 'ButtonDownFcn', 'MV_tool(''Step'')'); 	
	X = get(h_all_axes(i), 'xlim');
	Y = get(h_all_axes(i), 'ylim');
	set(fig, 'CurrentAxes', h_all_axes(i));
	htFrameNumbers(i) = text(X(2)*0.98, Y(2), num2str(getappdata(h_all_axes(i), 'CurrentImage')) ,...
		'Fontsize', textFontSize, 'color', [ 1 0.95 0], 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'visible', visibility);
end;
set(fig, 'CurrentAxes', h_axes);
set(0,'currentfigure', fig2);


handlesMV.CurrentAxes = h_axes;
handlesMV.ParentFigure = fig;
handlesMV.htFrameNumbers = htFrameNumbers;
set(handlesMV.Frame_Value_edit, 'String', getappdata(h_axes,'CurrentImage'));	

guidata(fig2,handlesMV);
Set_Current_Axes(h_axes);

% Draw faster and without flashes
set(fig, 'Closerequestfcn', [ old_CRF , ',MV_tool(''Close_Parent_Figure'');']);
set(fig, 'Renderer', 'zbuffer');
set(0, 'ShowHiddenHandles', 'On', 'CurrentFigure', fig);
set(gca,'Drawmode', 'Fast');

% store the figure's old infor within the fig's own userdata
set(fig, 'UserData', {fig2, old_WBDF, old_WBMF, old_WBUF, old_UserData,old_CRF, ...
		old_ToolEnables, old_ToolHandles, old_ToolStates });
set(fig, 'WindowButtonMotionFcn', '');  % entry function sets the WBMF


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Deactivate_View_Images(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

global GLOBAL_STOP_MOVIE 
GLOBAL_STOP_MOVIE = 2;

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');    
    hNewButton = gcbo;
    set(findobj('Tag', 'menuViewImages'),'checked', 'Off');
else
    hNewButton = varargin{1};
end;
    
% Reactivate other buttons
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure'),
    fig = get(fig, 'Parent');
end

hToolbar = findall(fig, 'type', 'uitoolbar');
if ~isempty(hToolbar)    hToolbar_Children = get(hToolbar, 'Children');
    set(findobj(hToolbar_Children,'Tag', 'figToolRotate3D'),'Enable', 'On');
    set(findobj(hToolbar_Children,'Tag', 'figToolZoomOut'),'Enable', 'On');
    set(findobj(hToolbar_Children,'Tag', 'figToolZoomIn'),'Enable', 'On');
end;

% Restore old BDFs
old_info= get(fig,'UserData');
set(fig, 'WindowButtonDownFcn', old_info{2});
set(fig, 'WindowButtonUpFcn', old_info{3});
set(fig, 'WindowButtonMotionFcn', old_info{4});

% Restore old Pointer and UserData
set(fig, 'UserData', old_info{5});
set(fig, 'CloseRequestFcn', old_info{6});
old_ToolEnables  = old_info{7};
old_ToolHandles = old_info{8};
old_ToolStates  = old_info{9};

fig2 = old_info{1};

handlesMV = guidata(fig2);
delete(handlesMV.htFrameNumbers);

for i = 1:length(handlesMV.Axes)
	set(findobj(handlesMV.Axes(i), 'Type', 'image'), 'ButtonDownFcn', '');
	
end;

for i = 1:length(old_ToolHandles)
	try
		set(old_ToolHandles(i), 'Enable', old_ToolEnables{i}, 'State', old_ToolStates{i});
	end;
end;
%LFG
%disable save_prefs tool button
SP = findobj(hToolbar_Children, 'Tag', 'figSavePrefsTool');
set(SP,'Enable','Off');

set(0, 'ShowHiddenHandles', 'Off');
try
    set(fig2, 'CloseRequestFcn', 'closereq'); 
    close(fig2); 
catch
	delete(fig2);
end;    


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Step(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

global GLOBAL_STOP_MOVIE;

if ~GLOBAL_STOP_MOVIE
	Stop_Movie;
end

fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
apply_all = get(handlesMV.Apply_radiobutton, 'Value');

if isempty(varargin);
	% mouse click, specify axis and function
	handlesMV.CurrentAxes = gca;
	selectiontype = get(handlesMV.ParentFigure, 'SelectionType');
	switch selectiontype 
		case 'normal'
			direction = 1;	
		case 'alt'
			direction = -1;
		case 'open'
			Play_Movie;
			return;
	end;
else
	% call from buttons
	direction = varargin{1};
end;

% specify single or all axes
CurrentAxes = handlesMV.CurrentAxes;
if apply_all
	CurrentAxes = handlesMV.Axes;
end;

for i = 1:length(CurrentAxes)
	current_frame = getappdata(CurrentAxes(i), 'CurrentImage');
	image_range   = getappdata(CurrentAxes(i), 'ImageRange');
	image_data    = getappdata(CurrentAxes(i), 'ImageData');

	if     (current_frame + direction) > image_range(2), current_frame = image_range(1); 
	elseif (current_frame + direction) < image_range(1), current_frame = image_range(2); 
	else                                                 current_frame = current_frame + direction; end;
	
	setappdata(CurrentAxes(i), 'CurrentImage', current_frame);
	set(handlesMV.htFrameNumbers(find(handlesMV.Axes == CurrentAxes(i))), 'String', num2str(current_frame));
	set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', squeeze(image_data(:,:,current_frame)));
	if (handlesMV.CurrentAxes==CurrentAxes(i))
		% if doing the single current axes, update the 
		set(handlesMV.Frame_Value_edit, 'String', num2str(current_frame));	
		Set_Current_Axes(CurrentAxes(i));
	end;
	figure(fig2);
end;
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Limit(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
apply_all = get(handlesMV.Apply_radiobutton, 'Value');

% call from buttons
direction = varargin{1};

% specify single or all axes
CurrentAxes = handlesMV.CurrentAxes;
if apply_all
	CurrentAxes = handlesMV.Axes;
end;

for i = 1:length(CurrentAxes)
	image_range   = getappdata(CurrentAxes(i), 'ImageRange');
	image_data    = getappdata(CurrentAxes(i), 'ImageData');
	
	if direction == 1
		current_frame = image_range(2);
	elseif direction == -1
		current_frame = image_range(1);
	end;
	
	setappdata( CurrentAxes(i), 'CurrentImage', current_frame);
	set(handlesMV.htFrameNumbers(find(handlesMV.Axes == CurrentAxes(i))), 'String', num2str(current_frame));
	set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', squeeze(image_data(:,:,current_frame)));
	if (handlesMV.CurrentAxes==CurrentAxes(i))
		set(handlesMV.Frame_Value_edit, 'String', num2str(current_frame));	
		Set_Current_Axes(CurrentAxes(i));
	end;
end;
figure(fig2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Set_Frame;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
apply_all = get(handlesMV.Apply_radiobutton, 'Value');

current_frame = str2num(get(handlesMV.Frame_Value_edit, 'String'));

% specify single or all axes
CurrentAxes = handlesMV.CurrentAxes;
if apply_all
	CurrentAxes = handlesMV.Axes;
end;

for i = 1:length(CurrentAxes)
	image_range   = getappdata(CurrentAxes(i), 'ImageRange');
	image_data    = getappdata(CurrentAxes(i), 'ImageData');
	
	% Error check
	if current_frame > image_range(2), current_frame = image_range(2); end;
	if current_frame < image_range(1), current_frame = image_range(1); end;
	
	setappdata( CurrentAxes(i), 'CurrentImage', current_frame);
	set(handlesMV.htFrameNumbers(find(handlesMV.Axes == CurrentAxes(i))), 'String', num2str(current_frame));
	set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', squeeze(image_data(:,:,current_frame)));
	if (handlesMV.CurrentAxes==CurrentAxes(i))
		set(handlesMV.Frame_Value_edit, 'String', num2str(current_frame));	
		Set_Current_Axes(CurrentAxes(i));
	end;
end;
figure(fig2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Set_Frame_Limit;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
CurrentAxes = handlesMV.CurrentAxes;
ImageRangeAll = getappdata(CurrentAxes, 'ImageRangeAll');
minFrame  = str2num(get(handlesMV.Min_Frame_edit, 'String'));
maxFrame  = str2num(get(handlesMV.Max_Frame_edit, 'String'));
currFrame = str2num(get(handlesMV.Frame_Value_edit, 'String'));
apply_all = get(handlesMV.Apply_radiobutton, 'Value');

if minFrame < ImageRangeAll(1), minFrame = ImageRangeAll(1); end;
if minFrame > currFrame       , minFrame = currFrame; end;

if maxFrame > ImageRangeAll(2), maxFrame = ImageRangeAll(2); end;
if maxFrame < currFrame       , maxFrame = currFrame; end;

% specify single or all axes
if apply_all
	CurrentAxes = handlesMV.Axes;
end;

for i = 1:length(CurrentAxes)
	setappdata(CurrentAxes(i), 'ImageRange', [minFrame maxFrame]);
	set(handlesMV.Min_Frame_edit, 'String', minFrame);
	set(handlesMV.Max_Frame_edit, 'String', maxFrame);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Reset_Frame_Limit;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
CurrentAxes = handlesMV.CurrentAxes;
apply_all = get(handlesMV.Apply_radiobutton, 'Value');

% specify single or all axes
if apply_all
	CurrentAxes = handlesMV.Axes;
end;

for i = 1:length(CurrentAxes)
	ImageRangeAll = getappdata(CurrentAxes(i), 'ImageRangeAll');
	setappdata(CurrentAxes(i), 'ImageRange',ImageRangeAll );
	set(handlesMV.Min_Frame_edit, 'String', num2str(ImageRangeAll(1)) );
	set(handlesMV.Max_Frame_edit, 'String', num2str(ImageRangeAll(2)) );
end;

Set_Current_Axes(handlesMV.CurrentAxes);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Play_Movie;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

global GLOBAL_STOP_MOVIE

fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
apply_all = get(handlesMV.Apply_radiobutton, 'Value');
frame_rate = str2num(get(handlesMV.Frame_Rate_edit, 'String'));

set([handlesMV.Reset_pushbutton, handlesMV.Min_Frame_edit, handlesMV.Max_Frame_edit, ...
		handlesMV.Frame_Value_edit, handlesMV.Rewind_pushbutton, handlesMV.Step_Rewind_pushbutton, ....
		handlesMV.Step_Forward_pushbutton, handlesMV.Forward_pushbutton, handlesMV.Play_pushbutton, ...
		handlesMV.Frame_Rate_edit, handlesMV.Make_Movie_pushbutton, ...
		handlesMV.Make_Mat_checkbox, handlesMV.Make_Avi_checkbox, handlesMV.Show_Frames_checkbox ], 'Enable', 'off');
		

% specify single or all axes
CurrentAxes = handlesMV.CurrentAxes;
if apply_all
	CurrentAxes = handlesMV.Axes;
end;

for i = 1:length(CurrentAxes)
	current_frame{i} = getappdata(CurrentAxes(i), 'CurrentImage');
	image_range{i}   = getappdata(CurrentAxes(i), 'ImageRange');
	image_data{i}    = getappdata(CurrentAxes(i), 'ImageData');
end;

GLOBAL_STOP_MOVIE = 0;
t = 0;
while ~GLOBAL_STOP_MOVIE
	tic
	for i = 1:length(CurrentAxes)
		direction = 1;
		if     (current_frame{i} + direction) > image_range{i}(2), current_frame{i} = image_range{i}(1); 
		elseif (current_frame{i} + direction) < image_range{i}(1), current_frame{i} = image_range{i}(2); 
		else                                                       current_frame{i} = current_frame{i} + direction; end;		
		set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', image_data{i}(:,:,current_frame{i}));
		set(handlesMV.htFrameNumbers(find(handlesMV.Axes == CurrentAxes(i))), 'String', num2str(current_frame{i}));
	end;
	drawnow;
	pause(t);
	if 1/toc > frame_rate, t = t+0.01; end;	
end;

% exit - update values for each of the axes in movie to correspond to last
% frame played
if (GLOBAL_STOP_MOVIE ~= 2)
	for i = 1:length(CurrentAxes)
		setappdata( CurrentAxes(i), 'CurrentImage', current_frame{i});		
		set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', image_data{i}(:,:,current_frame{i}));
		drawnow;
		if (handlesMV.CurrentAxes==CurrentAxes(i))
			% if doing the single current axes 
			set(handlesMV.Frame_Value_edit, 'String', num2str(current_frame{i}));	
			Set_Current_Axes(CurrentAxes(i));
		end;
	end;
	
	% Turn objects back on
	set([handlesMV.Reset_pushbutton, handlesMV.Min_Frame_edit, handlesMV.Max_Frame_edit, ...
			handlesMV.Frame_Value_edit, handlesMV.Rewind_pushbutton, handlesMV.Step_Rewind_pushbutton, ....
			handlesMV.Step_Forward_pushbutton, handlesMV.Forward_pushbutton, handlesMV.Play_pushbutton, ...
			handlesMV.Frame_Rate_edit, handlesMV.Make_Movie_pushbutton, ...
			handlesMV.Make_Mat_checkbox, handlesMV.Make_Avi_checkbox, handlesMV.Show_Frames_checkbox], 'Enable', 'On');
	figure(fig2);
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Make_Movie;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
CurrentAxes = handlesMV.CurrentAxes;
handle_for_movie = CurrentAxes;
apply_all = get(handlesMV.Apply_radiobutton, 'Value');
make_avi  = get(handlesMV.Make_Avi_checkbox, 'Value');
make_mat  = get(handlesMV.Make_Mat_checkbox, 'Value');
frame_rate = str2num(get(handlesMV.Frame_Rate_edit, 'String'));
minFrame   = str2num(get(handlesMV.Min_Frame_edit, 'String'));
maxFrame   = str2num(get(handlesMV.Max_Frame_edit, 'String'));

if make_avi | make_mat
    if str2num(version('-release')) > 12.1        
        [filename, pathname] = uiputfile( {'*.m;*.avi', 'Movie Files (*.m, *.avi)'}, ...
            'Save Movie As', 'M');
    else
        [filename, pathname] = uiputfile( {'*.m;*.avi', 'Movie Files (*.m, *.avi)'}, ...
            'Save Movie As');
    end;

	if isequal(filename,0) | isequal(pathname,0)
		% User hit cancel instead of ok
		return;
	end;

    filename = [pathname, filename];
	
	
	% Turn objects off while movie is made
	set([handlesMV.Reset_pushbutton, handlesMV.Min_Frame_edit, handlesMV.Max_Frame_edit, ...
		handlesMV.Frame_Value_edit, handlesMV.Rewind_pushbutton, handlesMV.Step_Rewind_pushbutton, ....
		handlesMV.Step_Forward_pushbutton, handlesMV.Forward_pushbutton, handlesMV.Play_pushbutton, ...
		handlesMV.Frame_Rate_edit, handlesMV.Make_Movie_pushbutton, handlesMV.Stop_pushbutton, ...
		handlesMV.Make_Avi_checkbox, handlesMV.Make_Mat_checkbox, handlesMV.Show_Frames_checkbox], ...
	'Enable', 'Off');	
else
	% do nothing!
	return;
end;
	
% if apply_all, make movie of the whole figure moving together.
if apply_all 
	CurrentAxes = handlesMV.Axes;
	handle_for_movie = handlesMV.ParentFigure;
end;

% collect info for each of the frames to be used.
for i = 1:length(CurrentAxes)
	image_range{i}   = getappdata(CurrentAxes(i), 'ImageRange');
	image_data{i}    = getappdata(CurrentAxes(i), 'ImageData');
	current_frame{i} = image_range{i}(1);
	
	if CurrentAxes(i)==handlesMV.CurrentAxes
		endFrame = image_range{i}(2); 
		iRef   = i;
	end;
end;

% play each frame; note that number of frames specified by the
% editable text boxes (ie the current axes frame limits) are used 
% to make movie - even if other windows have different number of 
% frames though each axes will start at their own beginning frame
stop_movie = 0;
counter = 1;
direction = 0;
while ~stop_movie
	for i = 1:length(CurrentAxes)
		if     (current_frame{i} + direction) > image_range{i}(2), current_frame{i} = image_range{i}(1); 
		elseif (current_frame{i} + direction) < image_range{i}(1), current_frame{i} = image_range{i}(2); 
		else                                                       current_frame{i} = current_frame{i} + direction; end;	
		set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', image_data{i}(:,:,current_frame{i}));
		set(handlesMV.htFrameNumbers(find(handlesMV.Axes == CurrentAxes(i))), 'String', num2str(current_frame{i}));
	end;
	drawnow;
	M(counter) = getframe(handle_for_movie);
	counter = counter + 1;
	direction = 1;
	% now determine if the movie is over: have played the last frame 
	% of the reference axes (current)
	if current_frame{iRef} == endFrame
		stop_movie = 1;
	end	
end;

if make_mat 
	f = [filename, '.mat'];
	save(f, 'M');
end;

compression = 'CinePak';
if isunix
	compression = 'None';
end;

if make_avi 
	f = [filename, '.avi'];
%	movie2avi(M, f, 'FPS', frame_rate, 'Compression', compression, 'Quality', 85);
	%movie2avi(M, f, 'FPS', frame_rate, 'Compression', 'none', 'Quality', 85);
        % Prepare the new file.
    vidObj = VideoWriter(f);
    
    vidObj.FrameRate = frame_rate;
    open(vidObj);
  
    writeVideo(vidObj,M);
    close(vidObj);
    %VideoWriter(M, f, 'FPS', frame_rate, 'Compression', 'none', 'Quality', 85);
end;	

% Turn objects back on
set([handlesMV.Reset_pushbutton, handlesMV.Min_Frame_edit, handlesMV.Max_Frame_edit, ...
		handlesMV.Frame_Value_edit, handlesMV.Rewind_pushbutton, handlesMV.Step_Rewind_pushbutton, ....
		handlesMV.Step_Forward_pushbutton, handlesMV.Forward_pushbutton, handlesMV.Play_pushbutton, ...
		handlesMV.Frame_Rate_edit, handlesMV.Make_Movie_pushbutton, handlesMV.Stop_pushbutton, ...
		handlesMV.Make_Avi_checkbox, handlesMV.Make_Mat_checkbox, handlesMV.Show_Frames_checkbox], ...
	'Enable', 'On');

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Stop_Movie;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

global GLOBAL_STOP_MOVIE
GLOBAL_STOP_MOVIE = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Show_Frames;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;
fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
visibility = get(handlesMV.Show_Frames_checkbox, 'Value');
if visibility, visibility = 'On' ;
else           visibility = 'Off'; end;
set(handlesMV.htFrameNumbers, 'visible', visibility);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Set_Current_Axes(currentaxes);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;
if isempty(currentaxes), currentaxes=gca; end;
fig2 = findobj('Tag', 'MV_figure');
handlesMV = guidata(fig2);
handlesMV.CurrentAxes = currentaxes;
image_range = getappdata(handlesMV.CurrentAxes, 'ImageRange');
set(handlesMV.Min_Frame_edit, 'string', num2str(image_range(1)));
set(handlesMV.Max_Frame_edit, 'string', num2str(image_range(2)));
guidata(fig2, handlesMV);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Menu_View_Images;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

hNewMenu = gcbo;
checked=  umtoggle(hNewMenu);
hNewButton = get(hNewMenu, 'userdata');

if ~checked
    % turn off button
    %Deactivate_Pan_Zoom(hNewButton);
    set(hNewMenu, 'Checked', 'off');
    set(hNewButton, 'State', 'off' );
else
    %Activate_Pan_Zoom(hNewButton);
    set(hNewMenu, 'Checked', 'on');
    set(hNewButton, 'State', 'on' );
end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Close_Parent_Figure;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to make sure that if parent figure is closed, 
% the ROI info and ROI Tool are closed too.
%global DB; if DB disp(['MV_Tool: ', Get_Current_Function]); end;

global GLOBAL_STOP_MOVIE 
GLOBAL_STOP_MOVIE = 2;

set(findobj('Tag', 'MV_figure'), 'Closerequestfcn', 'closereq');
try 
    close(findobj('Tag','MV_figure'));
end;


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