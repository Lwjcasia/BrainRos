// Generated by gencpp from file ford_msgs/GetSafeActionsRequest.msg
// DO NOT EDIT!


#ifndef FORD_MSGS_MESSAGE_GETSAFEACTIONSREQUEST_H
#define FORD_MSGS_MESSAGE_GETSAFEACTIONSREQUEST_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseStamped.h>

namespace ford_msgs
{
template <class ContainerAllocator>
struct GetSafeActionsRequest_
{
  typedef GetSafeActionsRequest_<ContainerAllocator> Type;

  GetSafeActionsRequest_()
    : start()
    , goal()  {
    }
  GetSafeActionsRequest_(const ContainerAllocator& _alloc)
    : start(_alloc)
    , goal(_alloc)  {
  (void)_alloc;
    }



   typedef  ::geometry_msgs::PoseStamped_<ContainerAllocator>  _start_type;
  _start_type start;

   typedef  ::geometry_msgs::PoseStamped_<ContainerAllocator>  _goal_type;
  _goal_type goal;





  typedef boost::shared_ptr< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> const> ConstPtr;

}; // struct GetSafeActionsRequest_

typedef ::ford_msgs::GetSafeActionsRequest_<std::allocator<void> > GetSafeActionsRequest;

typedef boost::shared_ptr< ::ford_msgs::GetSafeActionsRequest > GetSafeActionsRequestPtr;
typedef boost::shared_ptr< ::ford_msgs::GetSafeActionsRequest const> GetSafeActionsRequestConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator1> & lhs, const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator2> & rhs)
{
  return lhs.start == rhs.start &&
    lhs.goal == rhs.goal;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator1> & lhs, const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace ford_msgs

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsFixedSize< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "2fe3126bd5b2d56edd5005220333d4fd";
  }

  static const char* value(const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x2fe3126bd5b2d56eULL;
  static const uint64_t static_value2 = 0xdd5005220333d4fdULL;
};

template<class ContainerAllocator>
struct DataType< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ford_msgs/GetSafeActionsRequest";
  }

  static const char* value(const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
{
  static const char* value()
  {
    return "geometry_msgs/PoseStamped start\n"
"geometry_msgs/PoseStamped goal\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/PoseStamped\n"
"# A Pose with reference coordinate frame and timestamp\n"
"Header header\n"
"Pose pose\n"
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
"MSG: geometry_msgs/Pose\n"
"# A representation of pose in free space, composed of position and orientation. \n"
"Point position\n"
"Quaternion orientation\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Point\n"
"# This contains the position of a point in free space\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"\n"
"================================================================================\n"
"MSG: geometry_msgs/Quaternion\n"
"# This represents an orientation in free space in quaternion form.\n"
"\n"
"float64 x\n"
"float64 y\n"
"float64 z\n"
"float64 w\n"
;
  }

  static const char* value(const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.start);
      stream.next(m.goal);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct GetSafeActionsRequest_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ford_msgs::GetSafeActionsRequest_<ContainerAllocator>& v)
  {
    s << indent << "start: ";
    s << std::endl;
    Printer< ::geometry_msgs::PoseStamped_<ContainerAllocator> >::stream(s, indent + "  ", v.start);
    s << indent << "goal: ";
    s << std::endl;
    Printer< ::geometry_msgs::PoseStamped_<ContainerAllocator> >::stream(s, indent + "  ", v.goal);
  }
};

} // namespace message_operations
} // namespace ros

#endif // FORD_MSGS_MESSAGE_GETSAFEACTIONSREQUEST_H
