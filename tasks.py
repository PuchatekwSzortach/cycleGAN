"""
Module with invoke tasks
"""

import invoke

import host.invoke.host
import net.invoke.tests
import net.invoke.train
import net.invoke.visualize


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(host.invoke.host)
ns.add_collection(net.invoke.tests)
ns.add_collection(net.invoke.train)
ns.add_collection(net.invoke.visualize)
