#! /usr/bin/env python
# encoding: utf-8
# Thomas Nagy, 2006 (ita)

# look for 'meow' below
#import Options

# the following two variables are used by the target "waf dist"
VERSION='0.0.1'
APPNAME='leonardExample'

# these variables are mandatory ('/' are converted automatically)
srcdir = '.'
blddir = 'build'

def init():
	pass

def set_options(opt):
	# options provided in a script in a subdirectory named "src"
	opt.sub_options('src')

	# options provided by the modules
	opt.tool_options('compiler_cxx')

	opt.add_option('--leonard-library', default=False, help='Set the library path for leonard.')

	# custom options
	#opt.add_option('--exe', action='store_true', default=False, help='Execute the program after it is compiled')

def configure(conf):
	import Options
	# conf.env['CXX'] = Options.options.meow
	# CXX=g++-3.0 ./waf.py configure will use g++-3.0 instead of 'g++'
	conf.check_tool('compiler_cxx')

	## batched builds can be enabled by including the module optim_cc
	# conf.check_tool('batched_cc')

	conf.sub_config('src')

	conf.env['CXXFLAGS_MYPROG']='-O3'
	conf.env['LIBPATH_LEONARD']=Options.options.leonard_library
	conf.env['LIB_LEONARD']='leonard'
	conf.env['SOME_INSTALL_DIR']='/tmp/ahoy/lib/'


	conf.check_cfg(path='allegro-config', args='--cflags --libs', package='', uselib_store='ALLEGRO')
	# works in a procedural way, so the order of calls does matter
	#conf.check_tool(['KDE3'])

	return
	# testcase for variants, look below
	env = conf.env.copy()
	env.set_variant('debug')
	conf.set_env_name('debug', env)
	conf.setenv('debug')
	conf.env['CXXFLAGS'] = '-D_REENTRANT -DDBG_ENABLED -Wall -O0 -ggdb3 -ftemplate-depth-128'

def build(bld):
	# process subfolders from here
	bld.add_subdirs('src')

	## installing resources and files - call them under build(bld) or shutdown()
	## to trigger the glob, ad a star in the name
	## the functions are not called if not in install mode
	#bld.install_files('${PREFIX}', 'src/a2.h src/a1.h')
	bld.install_files('${PREFIX}/include', 'src/*.h')
	#install_as('${PREFIX}/dir/bar.png', 'foo.png')

	return
	# testcase for variants
	for obj in copy.copy(TaskGen.g_allobjs):
		new_obj = obj.clone('debug')

def shutdown():
	# command to execute
	cmd = "PATH=plugins:$PATH LD_LIBRARY_PATH=build/default/src/:$LD_LIBRARY_PATH build/default/src/testprogram"

	# if the custom option is set, execute the program
	import os, Options

	# in case if more people ask
	#if Options.commands['install']:
	#	try: os.popen("/sbin/ldconfig")
	#	except: pass

