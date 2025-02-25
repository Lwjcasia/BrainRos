// Generated by gencpp from file laser_cluster/Clusters.msg
// DO NOT EDIT!


#ifndef LASER_CLUSTER_MESSAGE_CLUSTERS_H
#define LASER_CLUSTER_MESSAGE_CLUSTERS_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Header.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/Point.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>

namespace laser_cluster
{
template <class ContainerAllocator>
struct Clusters_
{
  typedef Clusters_<ContainerAllocator> Type;

  Clusters_()
    : header()
    , labels()
    , counts()
    , mean_points()
    , max_points()
    , min_points()
    , pointclouds()
    , velocities()  {
    }
  Clusters_(const ContainerAllocator& _alloc)
    : header(_alloc)
    , labels(_alloc)
    , counts(_alloc)
    , mean_points(_alloc)
    , max_points(_alloc)
    , min_points(_alloc)
    , pointclouds(_alloc)
    , velocities(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Header_<ContainerAllocator>  _header_type;
  _header_type header;

   typedef std::vector<uint32_t, typename ContainerAllocator::template rebind<uint32_t>::other >  _labels_type;
  _labels_type labels;

   typedef std::vector<uint32_t, typename ContainerAllocator::template rebind<uint32_t>::other >  _counts_type;
  _counts_type counts;

   typedef std::vector< ::geometry_msgs::Point_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::Point_<ContainerAllocator> >::other >  _mean_points_type;
  _mean_points_type mean_points;

   typedef std::vector< ::geometry_msgs::Point_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::Point_<ContainerAllocator> >::other >  _max_points_type;
  _max_points_type max_points;

   typedef std::vector< ::geometry_msgs::Point_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::Point_<ContainerAllocator> >::other >  _min_points_type;
  _min_points_type min_points;

   typedef std::vector< ::sensor_msgs::PointCloud2_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::sensor_msgs::PointCloud2_<ContainerAllocator> >::other >  _pointclouds_type;
  _pointclouds_type pointclouds;

   typedef std::vector< ::geometry_msgs::Vector3_<ContainerAllocator> , typename ContainerAllocator::template rebind< ::geometry_msgs::Vector3_<ContainerAllocator> >::other >  _velocities_type;
  _velocities_type velocities;





  typedef boost::shared_ptr< ::laser_cluster::Clusters_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::laser_cluster::Clusters_<ContainerAllocator> const> ConstPtr;

}; // struct Clusters_

typedef ::laser_cluster::Clusters_<std::allocator<void> > Clusters;

typedef boost::shared_ptr< ::laser_cluster::Clusters > ClustersPtr;
typedef boost::shared_ptr< ::laser_cluster::Clusters const> ClustersConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::laser_cluster::Clusters_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::laser_cluster::Clusters_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::laser_cluster::Clusters_<ContainerAllocator1> & lhs, const ::laser_cluster::Clusters_<ContainerAllocator2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.labels == rhs.labels &&
    lhs.counts == rhs.counts &&
    lhs.mean_points == rhs.mean_points &&
    lhs.max_points == rhs.max_points &&
    lhs.min_points == rhs.min_points &&
    lhs.pointclouds == rhs.pointclouds &&
    lhs.velocities == rhs.velocities;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::laser_cluster::Clusters_<ContainerAllocator1> & lhs, const ::laser_cluster::Clusters_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace laser_cluster

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::laser_cluster::Clusters_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::laser_cluster::Clusters_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::laser_cluster::Clusters_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::laser_cluster::Clusters_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::laser_cluster::Clusters_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::laser_cluster::Clusters_<ContainerAllocator> const>
  : TrueType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::laser_cluster::Clusters_<ContainerAllocator> >
{
  static const char* value()
  {
    return "4df2e44fb76f60747b49f1ef14546917";
  }

  static const char* value(const ::laser_cluster::Clusters_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x4df2e44fb76f6074ULL;
  static const uint64_t static_value2 = 0x7b49f1ef14546917ULL;
};

template<class ContainerAllocator>
struct DataType< ::laser_cluster::Clusters_<ContainerAllocator> >
{
  static const char* value()
  {
    return "laser_cluster/Clusters";
  }

  static const char* value(const ::laser_cluster::Clusters_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::laser_cluster::Clusters_<ContainerAllocator> >
{
  static const char* value()
  {
    return "Header header\n"
"uint32[] labels\n"
"uint32[] counts\n"
"geometry_msgs/Point[] mean_points\n"
"geometry_msgs/Point[] max_points\n"
"geometry_msgs/Point[] min_points\n"
"sensor_msgs/PointCloud2[] pointclouds\n"
"geometry_msgs/Vector3[] velocities\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Header\n"
"# Standard metadata for higher-level stamped data types.\n"
"# This is generally used to communicate timestamped data \n"
"# in a particular coordinate frame.\n"
"# \n"
"# sequence ID: consecutively increasing ID \n"
"uint32 seq\n"
"#Two-integer timestamp that is expressed as:\n"
"# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')\n"
"# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')\n"
"# time-handling sugar is provided by the client library\n"
"time stamp\n"
"#Frame this data is associated with\n"
"string frame_id\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: sensor_msgs/PointCloud2\n"
"# This message holds a collection of N-dimensional points, which may\n"
"# contain additional information such as normals, intensity, etc. The\n"
"# point data is stored as a binary blob, its layout described by the\n"
"# contents of the \"fields\" array.\n"
"\n"
"# The point cloud data may be organized 2d (image-like) or 1d\n"
"# (unordered). Point clouds organized as 2d images may be produced by\n"
"# camera depth sensors such as stereo or time-of-flight.\n"
"\n"
"# Time of sensor data acquisition, and the coordinate frame ID (for 3d\n"
"# points).\n"
"Header header\n"
"\n"
"# 2D structure of the point cloud. If the cloud is unordered, height is\n"
"# 1 and width is the length of the point cloud.\n"
"uint32 height\n"
"uint32 width\n"
"\n"
"# Describes the channels and their layout in the binary data blob.\n"
"PointField[] fields\n"
"\n"
"bool    is_bigendian # Is this data bigendian?\n"
"uint32  point_step   # Length of a point in bytes\n"
"uint32  row_step     # Length of a row in bytes\n"
"uint8[] data         # Actual point data, size is (row_step*height)\n"
"\n"
"bool is_dense        # True if there are no invalid points\n"
"\n"
"================================================================================\n"
"MSG: sensor_msgs/PointField\n"
"# This message holds the description of one point entry in the\n"
"# PointCloud2 message format.\n"
"uint8 INT8    = 1\n"
"uint8 UINT8   = 2\n"
"uint8 INT16   = 3\n"
"uint8 UINT16  = 4\n"
"uint8 INT32   = 5\n"
"uint8 UINT32  = 6\n"
"uint8 FLOAT32 = 7\n"
"uint8 FLOAT64 = 8\n"
"\n"
"string name      # Name of field\n"
"uint32 offset    # Offset from start of point struct\n"
"uint8  datatype  # Datatype enumeration, see above\n"
"uint32 count     # How many elements in the field\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Vector3\n"
"# This represents a vector in free space. \n"
"# It is only meant to represent a direction. Therefore, it does not\n"
"# make sense to apply a translation to it (e.g., when applying a \n"
"# generic rigid transformation to a Vector3, tf2 will only apply the\n"
"# rotation). If you want your data to be translatable too, use the\n"
"# geometry_msgs/Point message instead.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
;
  }

  static const char* value(const ::laser_cluster::Clusters_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::laser_cluster::Clusters_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.header);
      stream.next(m.labels);
      stream.next(m.counts);
      stream.next(m.mean_points);
      stream.next(m.max_points);
      stream.next(m.min_points);
      stream.next(m.pointclouds);
      stream.next(m.velocities);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Clusters_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::laser_cluster::Clusters_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::laser_cluster::Clusters_<ContainerAllocator>& v)
  {
    s << indent << "header: ";
    s << std::endl;
    Printer< ::std_msgs::Header_<ContainerAllocator> >::stream(s, indent + "  ", v.header);
    s << indent << "labels[]" << std::endl;
    for (size_t i = 0; i < v.labels.size(); ++i)
    {
      s << indent << "  labels[" << i << "]: ";
      Printer<uint32_t>::stream(s, indent + "  ", v.labels[i]);
    }
    s << indent << "counts[]" << std::endl;
    for (size_t i = 0; i < v.counts.size(); ++i)
    {
      s << indent << "  counts[" << i << "]: ";
      Printer<uint32_t>::stream(s, indent + "  ", v.counts[i]);
    }
    s << indent << "mean_points[]" << std::endl;
    for (size_t i = 0; i < v.mean_points.size(); ++i)
    {
      s << indent << "  mean_points[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Point_<ContainerAllocator> >::stream(s, indent + "    ", v.mean_points[i]);
    }
    s << indent << "max_points[]" << std::endl;
    for (size_t i = 0; i < v.max_points.size(); ++i)
    {
      s << indent << "  max_points[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Point_<ContainerAllocator> >::stream(s, indent + "    ", v.max_points[i]);
    }
    s << indent << "min_points[]" << std::endl;
    for (size_t i = 0; i < v.min_points.size(); ++i)
    {
      s << indent << "  min_points[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Point_<ContainerAllocator> >::stream(s, indent + "    ", v.min_points[i]);
    }
    s << indent << "pointclouds[]" << std::endl;
    for (size_t i = 0; i < v.pointclouds.size(); ++i)
    {
      s << indent << "  pointclouds[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::sensor_msgs::PointCloud2_<ContainerAllocator> >::stream(s, indent + "    ", v.pointclouds[i]);
    }
    s << indent << "velocities[]" << std::endl;
    for (size_t i = 0; i < v.velocities.size(); ++i)
    {
      s << indent << "  velocities[" << i << "]: ";
      s << std::endl;
      s << indent;
      Printer< ::geometry_msgs::Vector3_<ContainerAllocator> >::stream(s, indent + "    ", v.velocities[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // LASER_CLUSTER_MESSAGE_CLUSTERS_H
