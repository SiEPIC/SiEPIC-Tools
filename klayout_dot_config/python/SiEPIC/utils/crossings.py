'''
Function to find crossings between paths and generate brakes on site, and helper functions and classes
Main Function: 
    insert_crossing(selected_paths, params, verbose= False)
 
Extra classes:
  iter2: A uselful iterator with several arbitrary functions that allow to alter it as it iterates, some very specific to pya.ObjInstPath objects.
  myPath: extended class for pya.Path with an 'order' function
    order: Given an input list of points, it return the ordered list based on when the point is found along the Path
  myBox: extended class for pya.Box
    move_to: move the Box to the input Point (from its center)

Version: 1.3 (Jan 16 2023)
Auhtor: Juan E. Villegas, August 2022
'''

# Enter your Python code here
from SiEPIC import _globals
from SiEPIC.utils import select_paths, get_layout_variables
import pya

# Returns true if a point p {pya.Point}, is in the segment  {list(pya.Point)}
# It supposes that the point is on the line that goes through the ends of the segment

def on_segment(segment, p,off=(0,0)):
    minx =  round(min((segment[0].x, segment[1].x))) #Rounding is needed when the paths are not snaped to the grid
    maxx =  round(max((segment[0].x, segment[1].x)))
    if p[0] >= minx+off[0] and p[0] <= maxx-off[0]:
      miny =  round(min((segment[0].y, segment[1].y)))
      maxy =  round(max((segment[0].y, segment[1].y)))
      if round(p[1]) >= miny+off[1] and round(p[1]) <= maxy-off[1]:
        return True
    return False

#Define a (infinte) line as Ax+By = C using two points on it
def myline(p1, p2): 
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C 

def det(V1, V2):
    return V1[0] * V2[1] - V1[1] * V2[0]

# Returns the point where line1 {line2} and line2 {line2}        
def line_intersect(line1, line2):
    div = det(line1[0:2], line2[0:2])
    Dx = det((line1[2],line1[1]), (line2[2],line2[1]) )
    Dy = det((line1[0],line1[2]), (line2[0],line2[2]) )
    if div == 0:
        return False
    x = Dx / div
    y = Dy / div
    return(x,y)


def path_intersection(path1, path2, xbox = None, radius = 0):
#DEBUG: Update so that it returns a list of points instead of Duples

    w = 0
    h = 0
    if isinstance(xbox, pya.Box):
      w = xbox.width();
      h = xbox.height();

    intersects = []   
    points_p1 = path1.each_point()
    p1_0 = next(points_p1)

    for  i in range(path1.num_points()-1):
      p1_1 = next(points_p1)
      line_01 = myline([p1_0.x, p1_0.y],[p1_1.x,p1_1.y])
      points_p2 = path2.each_point();
      p2_0 = next(points_p2)
      for  j in range(path2.num_points()-1):      
        
        p2_1 = next(points_p2)
        line_02 = myline([p2_0.x, p2_0.y],[p2_1.x,p2_1.y])
        
        x_center = line_intersect(line_01, line_02)
        
        if x_center:
          #print('%s: %s - %s'%(point,  on_segment((p2_0,p2_1), point), on_segment((p1_0,p1_1), point))  )
          if (line_01[0] == 0): #Line 1 is horizontal
              off_x = (int(w/2)+radius, 0)
              off_y = (0, int(h/2)+radius)
          else:
              off_x = (0, int(h/2)+radius)
              off_y = (int(w/2)+radius, 0)

          if on_segment((p1_0,p1_1), x_center,off_x) and on_segment((p2_0,p2_1), x_center,off_y) :
            intersects.append(x_center) #intersects.append(Point(x_center))
        p2_0 = p2_1
      p1_0 = p1_1
    return intersects


  
class iter2:
    """Iterator with prev and pop methods (among others)."""
    "  Very bad practice but we need them                  "
    
    def __init__(self, data):
        self.data = data
        self.index = -1

    def __iter__(self):
        return self

    def __prev__(self):
        if self.index <= -1:
            self.index = -1
        else:
            self.index = self.index - 1
        return self.data[self.index]
       
    def __next__(self):
        if self.index == len(self.data)-1:
            raise StopIteration
        self.index = self.index + 1
        return self.data[self.index]
    
    def prev(self):
      return self.__prev__()
    
    def set_index(self,index=-1):
      if index >= -1:
        self.index = index
        return True
      return False
    
    def assign(self,iter):
        return iter2(iter.data)

    def index(self):
        return self.index
        
    def insert(self, index, item):
        if isinstance(item, type(self.data[0])):
          self.data.insert(index, item)
        elif isinstance(item, list):
          for item_i in item:
            self.insert(index, item_i)
            index = index+1  
        return self.data[self.index]

    def pop(self, index):
        if index <= self.size():
          popel = self.data.pop(index)
          if index < self.index:
            self.index = self.index - 1
          return popel
        return
        
    def pop_and_del(self, index):
      if isinstance(self.data[index],pya.ObjectInstPath):
          popel = self.pop(index) 
          if not(popel.is_cell_inst()):
            popel.shape.delete()
          else:
            print('The element is a cell and not a path, cannot delete it.')
    
    def size(self):
      return len(self.data)

    def is_last(self):
      if self.index>= self.size():
          return True
      return False
    
    def get_data(self):
      return self.data


class myPath(pya.Path):
      def __init__(self,arg1, arg2=0):
            if isinstance(arg1,pya.Path):
              path = pya.Path.new(arg1)
            else:
              path = pya.Path.new(arg1, arg2)
            super(pya.Path, self).__init__()
            self.assign(path)
      
      def order(self,list_points):
            #Orders a list of points that are contained on a path, in the order in which they are found along it
            points = self.each_point()
            p0 = next(points)
            new_list = []
            for p1 in points:
              for c in list_points:
                if on_segment([p0, p1], c):
                  dist = p0.distance(pya.Point(c[0],c[1]))
                  new_list.append([dist, c])
              new_list.sort(key=lambda arg:arg[0])
              p0 = p1;
            data_out = [row[1] for row in new_list]
            return data_out
      
       
      def __sub__(self,bb):
            # This works for Manhattan paths only and for one point so far :(
                # It assumes that at most one point on a crossing lies within the BBox
                # Needs work to make it more general (jvillegas, Aug 2022)
                
            newp = []
            itp = self.each_point();
            
            def subs_point(p, op, bb):
              if(p.x == op.x): 
                
                if op.y>p.y:
                  p.y = bb.top 
                else:
                  p.y = bb.bottom 
                  
              elif p.y == op.y: 
                if p.x<op.x:
                  p.x = bb.right 
                else:
                  p.x = bb.left 
              return p
              
            for p in itp:
              if bb.contains(p):
                if len(newp)>0:
                  op = newp[-1]
                  subs_point(p,op,bb)
                  newp.append(p)   
                else:   
                  nextp = next(itp);
                  subs_point(p,nextp,bb)
                  newp.append(p)
                  if bb.contains(nextp):
                    subs_point(nextp,p,bb)
                  newp.append(nextp)
                   
              else:
                newp.append(p)
            self.points = newp  
            return self
        
     
      
def new_OIP(points, layer=0, _cell = 0, cv_index = 0): #uses obj as a reference for all the new instantiation    
        TECHNOLOGY, lv, ly, top_cell = get_layout_variables()
            
        newShapePath = ly.cell(_cell).shapes(layer).insert(pya.Path.new(points))         
        OIP_obj = pya.ObjectInstPath() 
        OIP_obj.layer = layer
        OIP_obj.cell_index = ly.cell(_cell).cell_index
        OIP_obj.top = _cell
        OIP_obj.cv_index = cv_index
        OIP_obj.shape = newShapePath
        return OIP_obj
 
 
class myBox(pya.Box):
        def __init__(self,arg1=0, arg2=0, arg3=0, arg4=0):
          super(pya.Box, self).__init__()
          if isinstance(arg1,pya.Box):
            self.assign(pya.Box.new(arg1))
          else:
            self.assign(pya.Box.new(arg1, arg2, arg3, arg4))
          
        def move_to(self,new_center):
          dx = int(new_center[0]-self.center().x)
          dy = int(new_center[1]-self.center().y)
          
          t = pya.Trans(dx,dy)
          self.assign(t*self)
        
# Break Path in the points listed in list_inter
# And returns a list of Obj Instantiations of the newly generated paths   
# obj is passed to match the layer and cell properties of the incoming instances, maybe this can be done cleaner

#DEBUG - Change list_inter to be a list of Points instead of a list of duples

def breakPath(path, list_inter, crossing_bbox, obj, verbose = False):
    
    bbox = myBox(crossing_bbox)
    #loop over the path segments to break them where they find a crossing (dont loop over the intersections)
    
    newp_points = []
    new_list = []
    
    old_bbox = myBox(0,0,0,0); 
    
    points = path.each_point()
    p0 = next(points)
    newp_points.append(p0)
    
    new_list_inter = path.order(list_inter);
    
    for i in range(path.num_points()-1):
        p1 = next(points)
        for center in new_list_inter:
            if on_segment((p0,p1), center):
                newp_points.append(pya.Point(center[0],center[1])) 
                bbox.move_to((center[0],center[1]));
                newPath = myPath(newp_points,path.width)
                newPath = newPath - bbox - old_bbox
                
               
                
                new_list.append(new_OIP(newPath, layer=obj.layer, _cell = obj.top, cv_index = obj.cv_index))
                old_bbox.assign(bbox)        
                if verbose: 
                  print('Appended path: ',new_list[-1].shape.path.	to_s())
                newp_points = []
                newp_points.append(pya.Point(center[0],center[1]))

        newp_points.append(p1)
        p0 = p1;
    newPath = myPath(newp_points,path.width)
    new_list.append(new_OIP(newPath - old_bbox, layer=obj.layer, _cell = obj.top, cv_index = obj.cv_index))
    if verbose:	 
      print('Appended path: ',new_list[-1].shape.path.to_s())
    return new_list


# Find crossings points between two points, and return the broken down pieces of the paths, leaving xcells placed on every crossing
def cross_2paths(oip_path1, oip_path2, xcell, offset = 0, origin = pya.Trans(0,0), verbose = False):
    TECHNOLOGY, lv, ly, cell = get_layout_variables()
    x_bbox = xcell.bbox()
    path1 = myPath(oip_path1.shape.path)
    path2 = myPath(oip_path2.shape.path)
  
    list_inter= path_intersection(path1,path2, x_bbox, offset)
    if not list_inter:
      return [], [], False
    
    if verbose:
      print("Found %s crossings between paths %s and %s"%(len(list_inter),oip_path1, oip_path2))
    
    '''
    Problem identified by Lukas Chrostowski:
      Since this function is called by scripts.path_to_waveguide2, 
      and that function has a transaction / commit, we cannot have a GUI inside here
      It would be nice if there was a separate function that checked to see
      if there were crossing paths, then brought up the GUI. Then the implementation
      would be split
        
    # Ask the user before insertion of waveguide crossings. Made global so that
    # it can be updated outside of utils.crossings
    
    global SiEPIC_crossings_UI_insertcrossing_flag

    if SiEPIC_crossings_UI_insertcrossing_flag:
        message_GUI = pya.QMessageBox()
        message_GUI.setStandardButtons(pya.QMessageBox.Yes | pya.QMessageBox.Cancel)
        message_GUI.setDefaultButton(pya.QMessageBox.Yes)
        message_GUI.setText("One or more intersections were found for a path.")
        message_GUI.setInformativeText("Do you want to create a crossing?")
        if (pya.QMessageBox_StandardButton(message_GUI.exec_()) == pya.QMessageBox.Cancel):
            SiEPIC_crossings_UI_insertcrossing_flag = True
            return [], [], False
        SiEPIC_crossings_UI_insertcrossing_flag = False
    '''
    
    new_path1 = (breakPath(path1, list_inter,x_bbox, oip_path1, verbose))
    new_path2 = (breakPath(path2, list_inter,x_bbox, oip_path2, verbose))
    
    for c in list_inter:
      t = pya.Trans(int(c[0]),int(c[1]))  
      cell.insert(pya.CellInstArray(xcell.cell_index(),t*origin))
    
    return new_path1, new_path2, True


## Main Fucntion, 
## Inserts a crossing at the crossed locations between pairs of paths and returns the a selection to the newly created paths
## and delets the previously selected ones.
## the process is n^2 to the number of paths selected, which increase by two everytime a path is found and restarts, making it 
## O(m!), where m is the number of paths that are left after inserting all crossings. Can it be made better? (most probably)
## It works only for self intersects provided that a second path also crosses separating the original self intersection (by luck). That can be extended
def insert_crossing(selected_paths, params = None, verbose = False):
    import time
    start = time.time()
    from SiEPIC.utils import select_paths, get_layout_variables
    TECHNOLOGY, lv, ly, top_cell = get_layout_variables()
    dbu = TECHNOLOGY['dbu']
    
    if params is None:
      return selected_paths
    
    xcell = None
    if 'crossing_cell' in params.keys():
      if 'crossing_library' in params.keys():
        xcell = top_cell.layout().create_cell(params['crossing_cell'], params['crossing_library'])
      else:
        xcell = top_cell.layout().create_cell(params['crossing_cell'], TECHNOLOGY['technology_name'])
    if xcell == None:
      if verbose:
        print("SiEPIC.utils.insert_crossing(): No available crossing cell in technology ", TECHNOLOGY['technology_name'])
      return selected_paths;
    
    global SiEPIC_crossings_UI_insertcrossing_flag 
    SiEPIC_crossings_UI_insertcrossing_flag = True

    test_instances = selected_paths
    iter_paths1 = iter2(test_instances)  
    
    for obj1 in iter_paths1:
      iter_paths2 = iter2(test_instances[iter_paths1.index+1::])
      #Iterate over the remaining paths to make pairwise crossing checks
      for obj2 in iter_paths2:
        crossing_origin = eval(params['crossing_offset'])
        new_paths1, new_paths2, check = cross_2paths(obj1, obj2, xcell ,offset=int(params['radius']/dbu),origin = pya.Trans(int(crossing_origin[0]/dbu),int(crossing_origin[1]/dbu) ), verbose=verbose)
        
        #if a crossing was inserted, the two paths are replaced by their broken down parts
        if check:
          idx_obj1 = iter_paths1.index
          idx_obj2 = iter_paths1.index+iter_paths2.index+1
          
          iter_paths1.pop_and_del(idx_obj2)         #remove the second oip 
          iter_paths1.insert(idx_obj2, new_paths2)  #insert the split parts of the path
          
          iter_paths1.pop_and_del(idx_obj1)         #then remove the first oip 
          iter_paths1.insert(idx_obj1, new_paths1)  #and insert the split parts of the path
          #iter_paths1.set_index(-1);               
          iter_paths1.prev()                        #Repeat the comaprison from the first of the newly split paths
          
          test_instances = iter_paths1.get_data()   #Update the outer iterator data to onclude the new paths in the exploration
          break
              
    end = time.time()
    if verbose:
      print("SiEPIC.utils.insert_crossing(): Finished finding crossing, ellapsed time: %s s"%(end - start))
      
    return iter_paths1.get_data()


def example(params=None, cell=None, snap=True, lv_commit=True, GUI=False, verbose=False, select_waveguides=False):
    TECHNOLOGY, lv, ly, top_cell = get_layout_variables()

    if not cell:
        cell = top_cell

    if lv_commit:
        lv.transaction("Insert crossings")

    if params is None:
        params = _globals.WG_GUI.get_parameters(GUI)
    if params is None:
        if verbose:
            print("SiEPIC.utils.insert_crossing() EXAMPLE: No parameters returned (user pressed Cancel); returning...")
        return
    if verbose:
        print("SiEPIC.utils.insert_crossing() EXAMPLE: params = %s" % params)

    selected_paths = select_paths(TECHNOLOGY['Waveguide'], cell, verbose=verbose)
    
    global SiEPIC_crossings_UI_insertcrossing_flag
    SiEPIC_crossings_UI_insertcrossing_flag = True #Flag to show the message to make crossings
      
    selected_paths = insert_crossing(selected_paths, params, verbose= True)
    lv.object_selection = selected_paths
    
example(verbose=True)