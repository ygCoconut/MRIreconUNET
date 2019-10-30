function PM_tool(varargin);
% function PM_tool(varargin);
% Function for point measurements on a set of images.  
% Use with imagesc or imagescn.
%
% Usage: PM_tool;
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

%disp(['PM Tool: Current Action: ', Action]);
switch Action
case 'New'
    Create_New_Button;

case 'Activate_Point_Tool'
    Activate_Point_Tool;
    
case 'Deactivate_Point_Tool'
    Deactivate_Point_Tool(varargin{2:end});
        
case 'Set_Current_Axes'
	Set_Current_Axes(varargin{2:end});
	
% case 'Measure'
%    Measure(varargin{2:end});
   
case 'Measure_Start' % Entry
   Measure_Start(varargin{2:end});

case 'Get_Coordinates' % Loop
   Get_Coordinates;

case 'Measure_End' % Exit
   Measure_End(varargin{2:end});

case 'Menu_Point_Tool'
    Menu_Point_Tool;
    
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
%disp('PM_tool: Create_New_Button');
fig = gcf;

% Find handle for current image toolbar and menubar
hToolbar = findall(fig, 'type', 'uitoolbar', 'Tag','FigureToolBar' );
hToolMenu = findall(fig, 'Label', '&Tools');

if ~isempty(hToolbar) & isempty(findobj(hToolbar, 'Tag', 'figPointTool'))
	hToolbar_Children = get(hToolbar, 'Children');
	
	% The default button size is 15 x 16 x 3. Create Button Image
   button_size_x= 16;
   button_image = NaN* zeros(15,button_size_x);
    
   f= [...
		   8    23    38    53    68    81    82    83    84    85    96   100   106   107 , ...
		   108   109   110   111   115   116   117   118   119   120   126   130   141   142 , ...
		   143   144   145   158   173   188   203   218   233 ...
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
      'OnCallback', 'PM_tool(''Activate_Point_Tool'')',...
      'OffCallback', 'PM_tool(''Deactivate_Point_Tool'')',...
      'Tag', 'figPointTool', ...
      'TooltipString', 'Point Measurement Tool',...
	  'Separator', separator, ...
      'UserData', [], ...
      'Enable', 'on');   
end;
  
% If the menubar exists, create menu item
if ~isempty(hToolMenu) & isempty(findobj(hToolMenu, 'Tag', 'menuPointTool'))
  hWindowLevelMenu = findobj(hToolMenu, 'Tag', 'menuWindowLevel');
  hPanZoomMenu     = findobj(hToolMenu, 'Tag', 'menuPanZoom');
  hROIToolMenu     = findobj(hToolMenu, 'Tag', 'menuROITool');
  hViewImageMenu   = findobj(hToolMenu, 'Tag', 'menuViewImages');
  hPointToolMenu   = findobj(hToolMenu, 'Tag', 'menuPointTool');
  hRotateToolMenu  = findobj(hToolMenu, 'Tag', 'menuRotateTool');
  hProfileToolMenu = findobj(hToolMenu, 'Tag', 'menuProfileTool');
  
  position = 9;
  separator = 'On';
  hMenus = [ hWindowLevelMenu, hPanZoomMenu,hROIToolMenu, hViewImageMenu, hRotateToolMenu, hProfileToolMenu ];
  if length(hMenus>0) 
	  position = position + length(hMenus);
	  separator = 'Off';
  end;
  
  hNewMenu = uimenu(hToolMenu,'Position', position);
  set(hNewMenu, 'Tag', 'menuPointTool','Label',...
      'Point Measurements',...
      'CallBack', 'PM_tool(''Menu_Point_Tool'')',...
      'Separator', separator,...
      'UserData', hNewButton...
  ); 
  
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Activate_Point_Tool(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('PM_tool:Activate_Point_Tool');

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');
    hNewButton = gcbo;
    set(findobj('Tag', 'menuPointTool'),'checked', 'on');
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
	PZ = findobj(hToolbar_Children, 'Tag', 'figPanZoom');
	RT = findobj(hToolbar_Children, 'Tag', 'figROITool');
	MV = findobj(hToolbar_Children, 'Tag', 'figViewImages');
	PM = findobj(hToolbar_Children, 'Tag', 'figPointTool');
	RotT = findobj(hToolbar_Children,'Tag', 'figRotateTool');
	Prof = findobj(hToolbar_Children, 'Tag', 'figProfileTool');
        
	
	old_ToolHandles  =     [Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,RotT,Prof];
	old_ToolEnables  = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,RotT,Prof], 'Enable');
	old_ToolStates   = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,RotT,Prof], 'State');
	
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
fig2_old = findobj('Tag', 'PM_figure');
% close the old WL figure to avoid conflicts
if ~isempty(fig2_old) close(fig2_old);end;

% open new figure
%LFG
fig2_file = 'PM_tool_figure.fig';
fig2 = openfig(fig2_file,'reuse');
optional_uicontrols = {'Measure_checkbox', 'Value'};
set(SP,'Userdata',{fig2, fig2_file, optional_uicontrols});


% Generate a structure of handles to pass to callbacks, and store it. 
handlesPM = guihandles(fig2);

close_str = [ 'hNewButton = findobj(''Tag'', ''figPointTool'');' ...
        ' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
        ' set(hNewButton, ''State'', ''off'' );' ...
        ' else,  ' ...
        ' PM_tool(''Deactivate_Point_Tool'',hNewButton);'...
        ' set(hNewButton, ''Value'', 0);',...
        ' end;' ];

set(fig2, 'Name', 'PM Tool',...
    'closerequestfcn', close_str);

old_pointer      = get(fig, 'Pointer');
old_pointer_data = get(fig, 'PointerShapeCData');
set(fig,'Pointer', 'fullcross'); %'cross'

% Record and store previous WBDF etc to restore state after PZ is done. 
old_WBDF = get(fig, 'WindowButtonDownFcn');
old_WBMF = get(fig, 'WindowButtonMotionFcn');
old_WBUF = get(fig, 'WindowButtonUpFcn');
old_UserData = get(fig, 'UserData');
old_CRF = get(fig, 'Closerequestfcn');
old_ShowHiddenHandles = get(0, 'ShowHiddenHandles');

% Store initial state of all axes in current figure for reset
h_all_axes = flipud(findobj(fig,'Type','Axes'));
h_axes = h_all_axes(1);

%set(h_all_axes, 'ButtonDownFcn', 'PM_tool(''Set_Current_Axes'', gca)');
set(fig, 'CurrentAxes', h_axes);

handlesPM.Axes = h_all_axes;
%for i = 1:length(h_all_axes)
%	set(findobj(h_all_axes(i), 'Type', 'image'), 'ButtonDownFcn', 'PM_tool(''Measure'', gca)'); 	
%end;
set(fig, 'WindowButtonDownFcn',  ['PM_tool(''Measure_Start'',',num2str(fig),')']);

handlesPM.CurrentAxes = h_axes;
handlesPM.ParentFigure = fig;

guidata(fig2,handlesPM);
Set_Current_Axes(h_axes);

h_axes = h_all_axes(end);
set(fig, 'CurrentAxes', h_axes);

% Draw faster and without flashes
set(fig, 'Closerequestfcn', [ old_CRF , ',PM_tool(''Close_Parent_Figure'')']);
set(fig, 'Renderer', 'zbuffer');
set(0, 'ShowHiddenHandles', 'On', 'CurrentFigure', fig);
set(gca,'Drawmode', 'Fast');

% store the figure's old infor within the fig's own userdata
set(fig, 'UserData', {fig2, old_WBDF, old_WBMF, old_WBUF, old_UserData,...
        old_pointer, old_pointer_data, old_CRF, ...
		old_ToolEnables,old_ToolHandles, old_ToolStates, ...
		old_ShowHiddenHandles});

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Deactivate_Point_Tool(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('PM_tool:Deactivate_Point_Tool');

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');    
    hNewButton = gcbo;
    set(findobj('Tag', 'menuPointTool'),'checked', 'Off');
else
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

myzoom('off');

% Restore old BDFs
old_info= get(fig,'UserData');

fig2 = old_info{1};
handlesPM = guidata(fig2);
for i = 1:length(handlesPM.Axes)
	set(findobj(handlesPM.Axes(i), 'Type', 'image'), 'ButtonDownFcn', ''); 	
end;

set(fig, 'WindowButtonDownFcn', old_info{2});
set(fig, 'WindowButtonUpFcn', old_info{3});
set(fig, 'WindowButtonMotionFcn', old_info{4});
% Restore old Pointer and UserData
set(fig, 'UserData', old_info{5});
set(fig, 'Pointer' , old_info{6});
set(fig, 'PointerShapeCData', old_info{7});
set(fig, 'CloseRequestFcn', old_info{8});
old_ToolEnables = old_info{9};
old_ToolHandles = old_info{10};
old_ToolStates  = old_info{11};
old_ShowHiddenHandles = old_info{12};

fig2 = old_info{1};
try
	set(fig2, 'CloseRequestFcn', 'closereq');
	close(fig2); 
catch
	delete(fig2);
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

set(0, 'ShowHiddenHandles', old_ShowHiddenHandles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Measure(currentaxes);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('PM_tool: Measure');

if isempty(currentaxes), currentaxes=gca; end;
fig2 = findobj('Tag', 'PM_figure');
handlesPM = guidata(fig2);
handlesPM.CurrentAxes = currentaxes;
guidata(fig2, handlesPM);
measure_all = get(handlesPM.Measure_checkbox, 'value');

% get position of last mouse click
currentpoint = get(currentaxes, 'CurrentPoint');

if measure_all 
	currentaxes = handlesPM.Axes;	
end;

s = [];
for i = 1:length(currentaxes)
	imagedata = get(findobj(currentaxes(i), 'Type', 'Image'), 'Cdata');
	s = strvcat(s,  Concat_To_String(...
	num2str(floor(currentpoint(1,1)), '%0.5g'), ...
	num2str(floor(currentpoint(1,2)), '%0.5g'), ...
	num2str( imagedata(floor(currentpoint(1,2)), floor(currentpoint(1,1))), '%0.5g')));
end;

set(handlesPM.Value_listbox, 'String', s);

figure(fig2);

%set(handlesPM.X_Value_text, 'String' , num2str(floor(currentpoint(1,1))));
%set(handlesPM.Y_Value_text, 'String' , num2str(floor(currentpoint(1,2))));
%set(handlesPM.Value_text,   'String' , num2str( imagedata(floor(currentpoint(1,2)), floor(currentpoint(1,1))), '%0.5g'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Measure_Start(fig);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('PM_tool: Measure_Start');
set(fig,'WindowButtonMotionFcn',['PM_tool(''Get_Coordinates'')']);
set(fig,'WindowButtonDownFcn'  ,['PM_tool(''Measure_End'',',num2str(fig),')']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Measure_End(fig);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('PM_tool: Measure_End');
set(fig,'WindowButtonMotionFcn','');
set(fig,'WindowButtonDownFcn'  ,['PM_tool(''Measure_Start'',',num2str(fig),')']);


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Get_Coordinates;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('PM_tool: Get_Coordinates');
fig2 = findobj('Tag', 'PM_figure');
handlesPM = guidata(fig2);

measure_all = get(handlesPM.Measure_checkbox, 'value');
currentaxes =get(handlesPM.ParentFigure,'CurrentAxes');
currentaxes_xlim = get(currentaxes, 'xlim');
currentaxes_ylim = get(currentaxes, 'ylim');
currentpoint =floor(get(currentaxes,'CurrentPoint'));

if measure_all 
	currentaxes = handlesPM.Axes;	
end;

% make sure that the current point is within the bounds of the current axis
if  ( currentpoint(1,1) >= currentaxes_xlim(1) & currentpoint(1,1) <= currentaxes_xlim(2) ) & ...
	( currentpoint(1,2) >= currentaxes_ylim(1) & currentpoint(1,2) <= currentaxes_ylim(2) ) 
	s = [];
	for i = 1:length(currentaxes)
		imagedata = get(findobj(currentaxes(i), 'Type', 'Image'), 'Cdata');
		s = strvcat(s,  Concat_To_String(...
			num2str(currentpoint(1,1), '%0.5g'), ...
			num2str(currentpoint(1,2), '%0.5g'), ...
			num2str( double(imagedata(floor(currentpoint(1,2)), floor(currentpoint(1,1)))), '%0.5g')));
	end;
	set(handlesPM.Value_listbox, 'String', s);
	figure(fig2);
end;




%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function outstr = Concat_To_String(str1, str2, str3);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%

min_size_1  = 4;
min_size_2  = 5;
min_size_3  = 8;

outstr = [ ...
		blanks( min_size_1 - length(str1)), str1 ,...
		blanks( min_size_2 - length(str2)), str2 ,...
		blanks( min_size_3 - length(str3)), str3 ,...
];

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Set_Current_Axes(currentaxes);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('PM_tool: Set_Current_Axes');

if isempty(currentaxes), currentaxes=gca; end;
fig2 = findobj('Tag', 'PM_figure');
handlesPM = guidata(fig2);
handlesPM.CurrentAxes = currentaxes;
guidata(fig2, handlesPM);

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Menu_Point_Tool;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('Menu_Point_Tool');

hNewMenu = gcbo;
checked=  umtoggle(hNewMenu);
hNewButton = get(hNewMenu, 'userdata');

if ~checked
    % turn off button
    %Deactivate_Point_Tool(hNewButton);
    set(hNewMenu, 'Checked', 'off');
    set(hNewButton, 'State', 'off' );
else
    %Activate_Point_Tool(hNewButton);
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
set(findobj('Tag', 'PM_figure'), 'Closerequestfcn', 'closereq');
try 
    close(findobj('Tag','PM_figure'));
end;



