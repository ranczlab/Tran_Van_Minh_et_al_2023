There are 4 Cre-lines for which we have host and presynaptic data:
	
	scnn1a
	rbp4
	ntsr1
	"gadOff"  (cre-off virus in gad2-cre)

Each file contains a Nx3 table array, where N is the number of mice in that cre-line.

For the presynaptic cells, the column "presynaptic" contains information about the location of each presynaptic neuron in that mouse. For hosts, the column "hosts" contains similar information.


Example
-------

If we load "scnn1a.mat", we have a variable "scnn1a" which is a 5x3 table (5 mice).

scnn1a.presynaptic{1} is a 2999x12 table containing information for each of the 2999 presynaptic neurons of the first scnn1a mouse.


scnn1a.presynaptic{1}.contralateral(j)  	- whether j-th presynaptic cell was contralateral (true) or ipsilateral (false)
scnn1a.presynaptic{1}.areaID(j) 			- (Allen) region ID of the j-th cell 


The format is similar for the "hosts_*" files, except the column of the table is labelled "hosts" instead of "presynaptic".
