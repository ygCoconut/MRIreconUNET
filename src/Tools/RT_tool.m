function RT_tool(varargin);
% function RT_tool(varargin);
% Function for rotations and flips of images
%
% Usage: RT_tool;
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

%(['RT Tool: Current Action: ', Action]);

switch Action
case 'New'
    Create_New_Button;

case 'Activate_Rotate_Tool'
    Activate_Rotate_Tool;
    
case 'Deactivate_Rotate_Tool'
    Deactivate_Rotate_Tool(varargin{2:end});
        
case 'Set_Current_Axes'
	Set_Current_Axes(varargin{2:end});
	
case 'Rotate_CW'
	Rotate_Images(0);

case 'Rotate_CCW'
	Rotate_Images(1);
	
case 'Flip_Horizontal'
	Flip_Images(0);

case 'Flip_Vertical' 
	Flip_Images(1);

case 'Menu_Rotate_Tool'
    Menu_Rotate_Tool;
    
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
%disp('RT_tool: Create_New_Button');
fig = gcf;

% Find handle for current image toolbar and menubar
hToolbar = findall(fig, 'type', 'uitoolbar', 'Tag','FigureToolBar' );
hToolMenu = findall(fig, 'Label', '&Tools');

if ~isempty(hToolbar) & isempty(findobj(hToolbar, 'Tag', 'figRotateTool'))
	hToolbar_Children = get(hToolbar, 'Children');
	
	% The default button size is 15 x 16 x 3. Create Button Image
   button_size_x= 16;
   button_image = NaN* zeros(15,button_size_x);
    
   f = [...
		   24    38    39    52    53    54    65    68    69    71    79    84    87   108   118   123   133,...
		   138   148   169   172   177   185   187   188   191   202   203   204   217   218   232 ...
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
      'OnCallback', 'RT_tool(''Activate_Rotate_Tool'')',...
      'OffCallback', 'RT_tool(''Deactivate_Rotate_Tool'')',...
      'Tag', 'figRotateTool', ...
      'TooltipString', 'Image Rotation Tool',...
	  'Separator', separator, ...
      'UserData', [], ...
      'Enable', 'on');   

end;

% If the menubar exists, create menu item
if ~isempty(hToolMenu) & isempty(findobj(hToolMenu, 'Tag', 'menuRotateTool'))
	
  hWindowLevelMenu = findobj(hToolMenu, 'Tag', 'menuWindowLevel');
  hPanZoomMenu     = findobj(hToolMenu, 'Tag', 'menuPanZoom');
  hROIToolMenu     = findobj(hToolMenu, 'Tag', 'menuROITool');
  hViewImageMenu   = findobj(hToolMenu, 'Tag', 'menuViewImages');
  hPointToolMenu   = findobj(hToolMenu, 'Tag', 'menuPointTool');
  hRotateToolMenu  = findobj(hToolMenu, 'Tag', 'menuRotateTool');
  hProfileToolMenu = findobj(hToolMenu, 'Tag', 'menuProfileTool');
	
  position = 9;
  separator = 'On';
  hMenus = [ hWindowLevelMenu, hPanZoomMenu, hROIToolMenu, hViewImageMenu, hPointToolMenu, hProfileToolMenu];

  if length(hMenus>0) 
	  position = position + length(hMenus);
	  separator = 'Off';
  end;
  RT = findobj('Tag', 'figRotateTool');
     
  hNewMenu = uimenu(hToolMenu,'Position', position);
  set(hNewMenu, 'Tag', 'menuRotateTool','Label',...
      'Rotate Flip Tool',...
      'CallBack', 'RT_tool(''Menu_Rotate_Tool'')',...
      'Separator', separator,...
      'UserData', hNewButton...
  ); 
  
end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Activate_Rotate_Tool(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('RT_tool:Activate_Point_Tool');

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');
    hNewButton = gcbo;
    set(findobj('Tag', 'menuRotateTool'),'checked', 'on');
else
    hNewButton = varargin{1};
end;

% allows for calls from buttons other than those in toolbar
fig = get(hNewButton, 'Parent');
if ~strcmp(get(fig, 'Type'), 'figure'),
    fig = get(fig, 'Parent');
end

% Deactivate zoom and rotate buttons
% will work even if there are no toolbars found
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
	PM = findobj(hToolbar_Children,'Tag', 'figPointTool');
	RotT = findobj(hToolbar_Children, 'Tag', 'figRotateTool');
	Prof = findobj(hToolbar_Children, 'Tag', 'figProfileTool');
	
	old_ToolHandles  =     [Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,PM,Prof];
	old_ToolEnables  = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,PM,Prof], 'Enable');
	old_ToolStates   = get([Rot3D, ZoomO, ZoomI,WL,PZ,RT,MV,PM,Prof], 'State');
	
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
% Start GUI
fig2_old = findobj('Tag', 'RT_figure');
% close the old WL figure to avoid conflicts
if ~isempty(fig2_old) close(fig2_old);end;

% open new figure
fig2_file = 'RT_tool_figure.fig';
fig2 = openfig(fig2_file,'reuse');
optional_uicontrols = {'Apply_checkbox', 'Value'};
set(SP,'Userdata',{fig2, fig2_file, optional_uicontrols});

% Generate a structure of handles to pass to callbacks, and store it. 
handlesRT = guihandles(fig2);

close_str = [ 'hNewButton = findobj(''Tag'', ''figRotateTool'');' ...
        ' if strcmp(get(hNewButton, ''Type''), ''uitoggletool''),'....
        ' set(hNewButton, ''State'', ''off'' );' ...
        ' else,  ' ...
        ' RT_tool(''Deactivate_Rotate_Tool'',hNewButton);'...
        ' set(hNewButton, ''Value'', 0);',...
        ' end;' ];

set(fig2, 'Name', 'RT Tool',...
    'closerequestfcn', close_str);

old_pointer      = get(fig, 'Pointer');
old_pointer_data = get(fig, 'PointerShapeCData');
%set(fig,'Pointer', 'fullcross'); %'cross'

% Record and store previous WBDF etc to restore state after PZ is done. 
old_WBDF = get(fig, 'WindowButtonDownFcn');
old_WBMF = get(fig, 'WindowButtonMotionFcn');
old_WBUF = get(fig, 'WindowButtonUpFcn');
old_UserData = get(fig, 'UserData');
old_CRF = get(fig, 'Closerequestfcn');

% Store initial state of all axes in current figure for reset
h_all_axes = flipud(findobj(fig,'Type','Axes'));
h_axes = h_all_axes(1);

%set(h_all_axes, 'ButtonDownFcn', 'RT_tool(''Set_Current_Axes'', gca)');
%for i = 1:length(h_all_axes)
%	set(findobj(h_all_axes(i), 'Type', 'image'), 'ButtonDownFcn', 'RT_tool(''Measure'', gca)'); 	
%end;

handlesRT.Axes = h_all_axes;
handlesRT.CurrentAxes = h_axes;
handlesRT.ParentFigure = fig;

% check for square images:
% if not square, then turn of image rotation (for now)
Im = get(findobj(handlesRT.CurrentAxes, 'Type', 'Image'), 'CData');
if(size(Im,1) ~= size(Im,2)), 
	set([handlesRT.Rotate_CCW_pushbutton, handlesRT.Rotate_CW_pushbutton], 'Enable', 'off');
end;

guidata(fig2,handlesRT);
Set_Current_Axes(h_axes);

%h_axes = h_all_axes(end);
set(fig, 'CurrentAxes', h_axes);
set(fig, 'WindowButtonDownFcn',  ['RT_tool(''Set_Current_Axes'')']);

% Draw faster and without flashes
set(fig, 'Closerequestfcn', [ old_CRF , ',RT_tool(''Close_Parent_Figure'')']);
set(fig, 'Renderer', 'zbuffer');
set(0, 'ShowHiddenHandles', 'On', 'CurrentFigure', fig);
set(gca,'Drawmode', 'Fast');

% store the figure's old infor within the fig's own userdata
set(fig, 'UserData', {fig2, old_WBDF, old_WBMF, old_WBUF, old_UserData,...
        old_pointer, old_pointer_data, old_CRF, ...
		old_ToolEnables,old_ToolHandles, old_ToolStates});

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Deactivate_Rotate_Tool(varargin);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('RT_tool:Deactivate_Point_Tool');

if nargin ==0
    set(0, 'ShowHiddenHandles', 'On');    
    hNewButton = gcbo;
    set(findobj('Tag', 'menuRotateTool'),'checked', 'Off');
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

% Restore old BDFs
old_info= get(fig,'UserData');

fig2 = old_info{1};
handlesRT = guidata(fig2);
for i = 1:length(handlesRT.Axes)
	set(findobj(handlesRT.Axes(i), 'Type', 'image'), 'ButtonDownFcn', ''); 	
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

fig2 = old_info{1};
try
	set(fig2, 'CloseRequestFcn', 'closereq');
	close(fig2); 
catch
	delete(fig2);
end;    

for i = 1:length(old_ToolHandles)
	try
		set(old_ToolHandles(i), 'Enable', old_ToolEnables{i}, 'State', old_ToolStates{i});	catc
	end;
end;
%LFG
%disable save_prefs tool button
SP = findobj(hToolbar_Children, 'Tag', 'figSavePrefsTool');
set(SP,'Enable','Off');

set(0, 'ShowHiddenHandles', 'Off');


%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Flip_Images(direction);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp(['RT_tool: Flip ', num2str(direction)]);

fig2 = findobj('Tag', 'RT_figure');
handlesRT = guidata(fig2);
%guidata(fig2, handlesRT);
apply_all = get(handlesRT.Apply_checkbox, 'value');

% specify single or all axes
CurrentAxes = handlesRT.CurrentAxes;
if apply_all, CurrentAxes = handlesRT.Axes; end;

for i = 1:length(CurrentAxes)
	% flip CData of the current image, 
	% change the xlims and ylims dependening on the call
	% if a 4-D array, flip all the other data in memory
	if isappdata(CurrentAxes(i), 'CurrentImage')
		image_data = getappdata(CurrentAxes(i), 'ImageData');	
		current_image = getappdata(CurrentAxes(i), 'CurrentImage');
		dim4 = 1;
	else
		image_data = get(findobj(CurrentAxes(i), 'Type', 'Image'), 'CData');		
		current_image = 1;
		dim4 = 0;
	end;
	
	xlims = get(CurrentAxes(i), 'Xlim');
	ylims = get(CurrentAxes(i), 'Ylim');
	im_size  = [size(image_data,1), size(image_data,2)];
	
	if direction  
		%disp('flip Vertical')
		for j =1:size(image_data,3)
			image_data(:,:,j) = flipud(image_data(:,:,j));
		end
		ylims = [ (im_size(1) ) - (ylims(2)-0.5) , im_size(1)  - (ylims(1)-0.5)] + 0.5;
	else 
		%disp('flip Horizontal')	
		for j =1:size(image_data,3)
			image_data(:,:,j) = fliplr(image_data(:,:,j));
		end
		xlims = [ (im_size(2) ) - (xlims(2)-0.5), (im_size(2) ) - (xlims(1)-0.5)] + 0.5;		
	end;

	if dim4, 
		setappdata(CurrentAxes(i), 'ImageData', image_data);
	end;
	
	set(CurrentAxes(i), 'Xlim', xlims, 'Ylim', ylims);
	set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', squeeze(image_data(:,:,current_image)));

end;
figure(fig2);
		
		
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Rotate_Images(direction);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp(['RT_tool: Flip ', num2str(direction)]);

fig2 = findobj('Tag', 'RT_figure');
handlesRT = guidata(fig2);
guidata(fig2, handlesRT);
apply_all = get(handlesRT.Apply_checkbox, 'value');

% specify single or all axes
CurrentAxes = handlesRT.CurrentAxes;
if apply_all, CurrentAxes = handlesRT.Axes; end;

for i = 1:length(CurrentAxes)
	% flip CData of the current image, 
	% change the xlims and ylims dependening on the call
	% if a 4-D array, flip all the other data in memory
	dim4 = 1;
	current_image = getappdata(CurrentAxes(i), 'CurrentImage');
	if ~isempty(current_image)
		image_data = getappdata(CurrentAxes(i), 'ImageData');	
	else
		image_data = get(findobj(CurrentAxes(i), 'Type', 'Image'), 'CData');		
		current_image = 1;
		dim4 = 0;
	end;
	
	im_size = ( size(image_data,1) + 1)/2;
	xlims = get(CurrentAxes(i), 'Xlim') - im_size;
	ylims = get(CurrentAxes(i), 'Ylim') - im_size;
	temp = xlims;

	if direction  % Rotate CCW
		for j =1:size(image_data,3)
			image_data(:,:,j) = flipud(permute(image_data(:,:,j), [ 2 1]));
		end			
		xlims =        ylims  + im_size;
		ylims =  sort(-1*temp + im_size);	
	else % Rorate CW	
		for j =1:size(image_data,3)
			image_data(:,:,j) = fliplr(permute(image_data(:,:,j), [ 2 1]));
		end		
		xlims = sort(-1*ylims + im_size);
		ylims =         temp  + im_size;
	end;

	if dim4, 
		setappdata(CurrentAxes(i), 'ImageData', image_data);
	end;
	
	set(CurrentAxes(i), 'Xlim', xlims, 'Ylim', ylims);
	set(findobj(CurrentAxes(i), 'Type', 'image'), 'CData', squeeze(image_data(:,:,current_image)));
	figure(fig2);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Set_Current_Axes(currentaxes);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('RT_tool: Set_Current_Axes');
if (nargin == 0), currentaxes = gca; end;
if isempty(currentaxes), currentaxes=gca; end;

fig2 = findobj('Tag', 'RT_figure');
handlesRT = guidata(fig2);
handlesRT.CurrentAxes = currentaxes;
guidata(fig2,handlesRT );

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
function Menu_Rotate_Tool;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%disp('RT_tool: Menu_Rotate_Tool');

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
%disp('RT_tool: Close_Parent_Figure');
set(findobj('Tag', 'RT_figure'), 'Closerequestfcn', 'closereq');
try 
    close(findobj('Tag','RT_figure'));
end;



