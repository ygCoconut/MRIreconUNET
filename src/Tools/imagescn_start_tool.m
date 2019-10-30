function imagescn_start_tool(varargin);

if isempty(varargin) 
   Action = 'New';
else
   Action = varargin{1};  
end

switch Action
	
	case 'New';
		Create_figure;
		
	case 'Browse'
		Browse;
		% Browse for files here
		
	case 'Load_Data'
		Load_Data;
		
	
otherwise
	disp(['Unimplemented Functionality: ', Action]);
	
end;

function Create_figure;
old_figs = findobj('Tag', 'Imagescn_Start_figure');
close(old_figs);
openfig('Imagescn_Start_figure');
	
function Browse;
% open gui
[fname, pname] = uigetfile('*.mat', 'Choose a mat file:')

if ~isequal(fname,0) & ~isequal(pname,0)
	start_fig = findobj('Tag', 'Imagescn_Start_figure');
	handles = guihandles(start_fig);
	set(handles.Filename_edit, 'String', [pname, fname]);
	guidata(start_fig, handles);
end

function Load_Data;

start_fig = findobj('Tag', 'Imagescn_Start_figure')

handles = guidata(start_fig);
filename = get(handles.Filename_edit, 'String');

cmin = get(handles.Cmin_edit, 'String');
cmax = get(handles.Cmax_edit, 'String');
if isempty(cmin) | isempty(cmax) 
	WL = [];
else
	try 
		cmin = str2num(cmin);
		cmax = str2num(cmax);
	catch
		disp(['Non numeric color range!!!!']);
		WL = [];
	end;
	WL = [cmin cmax];
end;

row = get(handles.Row_edit, 'String');
col = get(handles.Col_edit, 'String');
if isempty(row) | isempty(col) 
	RC = [];
else
	try 
		row = str2num(row);
		col = str2num(col);
	catch
		disp(['Non numeric layout!!!!']);
		RC = [];
	end;
	RC = [row col];
end;

im_width = get(handles.Width_edit, 'String');
if isempty(im_width)
	im_width = [];
else
	try 
		im_width = str2num(im_width);
	catch
		disp(['Non numeric image width!!!!']);
		im_width = [];
	end;
end;	
	
temp_dim = get(handles.Time_edit, 'String')
if isempty(temp_dim)
	temp_dim = [];
else
	try 
		temp_dim = str2num(temp_dim);
	catch
		disp(['Non numeric temporal dimension!!!!']);
		temp_dim = [];
	end;
end;

if ~exist(filename,'file') 
	disp('File Does not Exist!!!!');
	return;
end;

temp = load(filename);
fields = fieldnames(temp);
fields = fields{1};

if isempty(temp_dim)
	imagescn(temp.(fields), WL, RC, im_width);
else
	imagescn(temp.(fields), WL, RC, im_width, temp_dim);
end;
close(start_fig);