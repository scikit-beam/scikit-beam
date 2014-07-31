# Copyright (c) Brookhaven National Lab 2O14
# All rights reserved
# BSD License
# See LICENSE for full text
# @author:  Li Li (lili@bnl.gov)
# created on 07/20/2014




class ParameterBase(object):
    """
    base class to save data structure 
    for each fitting parameter
    """
    def __init__(self):
        self.val = None
        self.min = None
        self.max = None
        return
    

class Parameters(object):
    
    def __init__(self):
        self.p_dict = {}
        return

    def add(self, **kwgs):
        if kwgs.has_key('name'):
            self.p_dict[kwgs['name']] = ParameterBase()
            
            if kwgs.has_key('val'): 
                self.p_dict[kwgs['name']].val = kwgs['val']
                
            if kwgs.has_key('min'): 
                self.p_dict[kwgs['name']].min = kwgs['min']
            
            if kwgs.has_key('max'): 
                self.p_dict[kwgs['name']].max = kwgs['max']
                
        else:
            print "please define parameter name first."
            print "please define parameters as %s, %s, %s, %s" \
            %('name', 'val', 'min', 'max')   
            
        return
    
    
    #def __setattr__(self, name, value):
    #    super(Parameters, self).__setattr__(name, value)
       
    
    def __getitem__(self, name):
        return self.p_dict[name]
    
    
    def all(self):
        return self.p_dict
    
    
    
    
    